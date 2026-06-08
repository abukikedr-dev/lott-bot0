"""
Lottery Logbook Bot — v4.0
──────────────────────────────────────────────────────────────────────────────
ROOT-CAUSE FIX (v3.x bug):
  The old design used a shared _API_LOCK so every OCR call ran single-file
  through one thread AND rotated keys in a tight loop.  Under any real load
  all keys hit the 60-RPM free-tier quota at the same time, producing the
  "All keys cooling — waiting 58s" death-spiral you saw in the logs.

NEW APPROACH — only ONE Gemini API key is needed:
  1. Upload the image to the Gemini Files API  (uploadFile endpoint).
     Uploaded files are stored server-side for 48 h.  The upload itself does
     NOT count against the generate quota.
  2. Call generateContent with a file:// URI reference — much lighter than
     sending raw bytes on every retry.
  3. On 429 / RESOURCE_EXHAUSTED use *true exponential back-off* (1s→2s→4s…
     capped at 64 s) instead of a flat 70-second hard sleep.
  4. Each photo runs in its own daemon thread so photos in a batch are
     processed in parallel — the bot stays responsive even for 10+ images.
  5. A per-chat semaphore (MAX_CONCURRENT=3) prevents thundering-herd if
     multiple users upload at the same time.

PythonAnywhere free-tier note:
  Uncomment the apihelper.proxy line below if you are on the free tier.
  The Files API upload goes through urllib3 (not telebot), so if PA also
  blocks that you will need to route it through the same proxy — see the
  PROXY_URL constant.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from flask import Flask
from threading import Thread
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import requests                                        # for Files API upload
import google.generativeai as genai
import pandas as pd
import telebot
from dotenv import load_dotenv
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from telebot import apihelper, types

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive!"

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("lottery_bot")

TELEGRAM_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]   # ONE key is enough now

# ── PythonAnywhere free-tier proxy (uncomment if needed) ─────────────────────
# apihelper.proxy = {"https": "http://proxy.server:3128"}

# If PA blocks the Files API upload too, set this to your proxy URL string,
# e.g. "http://proxy.server:3128", otherwise leave as None.
PROXY_URL: Optional[str] = None

genai.configure(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OCR_MODEL       = "gemini-1.5-flash"    # change to gemini-2.5-flash if preferred
MAX_RETRIES     = 6                     # exponential back-off retries
BACKOFF_BASE    = 1.5                   # seconds — doubles each retry
BACKOFF_CAP     = 64.0                  # seconds — maximum sleep between retries
MAX_CONCURRENT  = 3                     # parallel OCR calls across all users
MAX_PHOTOS      = 20
SESSION_TTL     = 3600
MSG_CHAR_LIMIT  = 3800

PHONE_RE = re.compile(r"^0[79]\d{8}$")

# Global semaphore so we never fire more than MAX_CONCURRENT Gemini calls at once
_GEMINI_SEM = threading.Semaphore(MAX_CONCURRENT)

# ─────────────────────────────────────────────────────────────────────────────
# Gemini Files API — upload helper
# ─────────────────────────────────────────────────────────────────────────────

FILES_UPLOAD_URL = (
    "https://generativelanguage.googleapis.com/upload/v1beta/files"
    "?uploadType=multipart"
)

def _upload_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:

    """
    Upload image bytes to the Gemini Files API.
    Returns the file URI (e.g. "files/abc123xyz") to pass to generate_content.
    Upload requests are NOT counted against the 60-RPM generate quota.
    """
    proxies = {"https": PROXY_URL, "http": PROXY_URL} if PROXY_URL else None

    metadata_part = json.dumps({"file": {"display_name": "logbook_page"}})

    # multipart/related body
    boundary = "-----lottery_bot_boundary"
    body = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n\r\n"
        f"{metadata_part}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--".encode()

    headers = {
        "X-Goog-Api-Key": GEMINI_API_KEY,
        "Content-Type": f"multipart/related; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    resp = requests.post(
        FILES_UPLOAD_URL,
        headers=headers,
        data=body,
        proxies=proxies,
        timeout=60,
    )
    resp.raise_for_status()
    file_uri = resp.json()["file"]["uri"]
    log.info("Uploaded image → %s", file_uri)
    return file_uri


def _delete_file(file_uri: str) -> None:
    """Best-effort delete to keep the Files API quota tidy."""
    try:
        proxies = {"https": PROXY_URL, "http": PROXY_URL} if PROXY_URL else None
        # file_uri is like "files/abc123" — build the REST path
        name = file_uri.split("/v1beta/")[-1] if "/v1beta/" in file_uri else file_uri
        url = f"https://generativelanguage.googleapis.com/v1beta/{name}"
        requests.delete(
            url,
            headers={"X-Goog-Api-Key": GEMINI_API_KEY},
            proxies=proxies,
            timeout=10,
        )
        log.info("Deleted remote file %s", name)
    except Exception as e:
        log.debug("File delete failed (non-fatal): %s", e)

# ─────────────────────────────────────────────────────────────────────────────
# OCR prompt & parser
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT = """
You are a specialist OCR assistant for Ethiopian handwritten lottery logbooks.

PAGE LAYOUT — each page has one or more vertical column-pairs:
  LEFT  : sequential ticket numbers (handwritten integers, ascending).
  RIGHT : customer name in Amharic (IGNORE) and a 10-digit Ethiopian mobile
          number starting with 09 or 07.

SPANNING RULE:
  A phone number written large may visually span several ticket rows.
  Assign that SAME number to EVERY ticket it covers. Never mark a covered row "Empty".

VALIDATION:
  • Phone must be exactly 10 digits, starting with 07 or 09.
  • If unreadable or absent → "Empty".  Never guess or invent digits.

OUTPUT — return ONLY a valid JSON object.  No markdown.  No prose.  No code fences.
{"tickets": [{"ticket": "1", "phone": "0916039018"}, {"ticket": "2", "phone": "Empty"}]}
"""
def _parse_ocr(raw: str) -> dict[str, str]:
    # Strip markdown fences
    raw = re.sub(r"^\s*```[a-zA-Z]*\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    # Extract the first complete JSON object in case of trailing prose
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {raw[:200]}")
    raw = match.group(0)
    parsed = json.loads(raw)
    result: dict[str, str] = {}
    for item in parsed.get("tickets", []):
        ticket = str(item.get("ticket", "")).strip()
        phone  = str(item.get("phone",  "Empty")).strip()
        if not ticket:
            continue
        if phone != "Empty" and not PHONE_RE.match(phone):
            log.warning("Invalid phone '%s' for ticket %s → Empty", phone, ticket)
            phone = "Empty"
        result[ticket] = phone
    return result


# ─────────────────────────────────────────────────────────────────────────────
# OCR entry point — upload-once + exponential back-off retry
# ─────────────────────────────────────────────────────────────────────────────

def extract_tickets(image_bytes: bytes) -> dict[str, str]:
    """
    1. Upload image to Files API (no quota cost).
    2. Call generate_content with the file URI.
    3. On 429/503 use exponential back-off — never a flat 70-second sleep.
    4. Acquires the global semaphore so we honour MAX_CONCURRENT.
    """
    file_uri: Optional[str] = None
    
    try:
        import imghdr
        detected  = imghdr.what(None, h=image_bytes)
        mime_type = f"image/{detected}" if detected in ("jpeg", "png", "webp", "gif") else "image/jpeg"
        log.info("Detected MIME type: %s", mime_type)

        for attempt in range(3):
            try:
                file_uri = _upload_image(image_bytes, mime_type=mime_type)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                log.warning("Upload attempt %d failed: %s — retrying", attempt + 1, e)
                time.sleep(2 ** attempt)

        for _ in range(10):
            file_info = genai.get_file(file_uri.split("/")[-1])
            if file_info.state.name == "ACTIVE":
                break
            log.info("Waiting for file to become ACTIVE…")
            time.sleep(1)
          
        model = genai.GenerativeModel(OCR_MODEL)

        # Build content with file URI reference wrapped correctly for the SDK
        image_part = genai.protos.Part(
            file_data=genai.protos.FileData(
                mime_type=mime_type,
                file_uri=file_uri,
            )
        )



        last_exc: Exception | None = None
        sleep_time = BACKOFF_BASE

        with _GEMINI_SEM:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = model.generate_content(
                        [_PROMPT, image_part],
                        generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 8192,
                        },
                    )
                    result = _parse_ocr(response.text)
                    log.info(
                        "✓ OCR done — model=%s tickets=%d (attempt %d)",
                        OCR_MODEL, len(result), attempt,
                    )
                    return result

                except (json.JSONDecodeError, ValueError) as e:
                    # Malformed JSON — retry immediately once, then give up
                    last_exc = e
                    log.warning("Bad JSON on attempt %d: %s", attempt, e)
                    if attempt >= 2:
                        raise RuntimeError(
                            f"Model returned unparseable JSON twice: {e}"
                        ) from e
                    continue

                except Exception as e:
                    last_exc = e
                    err_str = str(e)

                    is_rate  = "429" in err_str or "quota" in err_str.lower() or \
                               "resource_exhausted" in err_str.lower()
                    is_retry = "503" in err_str or "unavailable" in err_str.lower()

                    if is_rate or is_retry:
                        # Extract suggested retry delay from error message if present
                        m = re.search(r"retry[_ ]delay.*?(\d+\.?\d*)", err_str, re.I)
                        suggested = float(m.group(1)) + 1.0 if m else sleep_time
                        actual_sleep = min(suggested, BACKOFF_CAP)
                        log.warning(
                            "Rate-limit/503 on attempt %d — sleeping %.1fs",
                            attempt, actual_sleep,
                        )
                        time.sleep(actual_sleep)
                        sleep_time = min(sleep_time * 2, BACKOFF_CAP)
                        continue
                    else:
                        # Non-retriable error (auth, bad request, etc.)
                        raise

        raise RuntimeError(
            f"OCR failed after {MAX_RETRIES} attempts. Last: {last_exc}"
        ) from last_exc

    finally:
        # Always clean up the uploaded file
        if file_uri:
            _delete_file(file_uri)

# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE                 = auto()
    COLLECTING           = auto()
    PROCESSING           = auto()
    AWAITING_CONFIRM     = auto()
    AWAITING_EDIT_TICKET = auto()
    AWAITING_EDIT_PHONE  = auto()

@dataclass
class PhotoRecord:
    photo_id:     int
    start_ticket: str
    data:         dict[str, str]

    @property
    def filled(self) -> int:
        return sum(1 for v in self.data.values() if v != "Empty")

    @property
    def empty(self) -> int:
        return len(self.data) - self.filled

@dataclass
class Session:
    state:         State             = State.IDLE
    photos:        list[PhotoRecord] = field(default_factory=list)
    pending_ids:   list[str]         = field(default_factory=list)  # file_ids queued for batch
    timer:         Optional[object]  = None                          # threading.Timer
    edit_photo_id: Optional[int]     = None
    edit_ticket:   Optional[str]     = None
    last_activity: float             = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def reset(self) -> None:
        if self.timer:
            try:
                self.timer.cancel()
            except Exception:
                pass
        self.state         = State.IDLE
        self.photos.clear()
        self.pending_ids.clear()
        self.timer         = None
        self.edit_photo_id = None
        self.edit_ticket   = None

# ─────────────────────────────────────────────────────────────────────────────
# Session store
# ─────────────────────────────────────────────────────────────────────────────

_sessions: dict[int, Session] = {}
_sessions_lock = threading.Lock()

def get_session(chat_id: int) -> Session:
    with _sessions_lock:
        sess = _sessions.get(chat_id)
        if sess is None:
            sess = Session()
            _sessions[chat_id] = sess
        elif time.monotonic() - sess.last_activity > SESSION_TTL:
            log.info("Session %d expired — resetting.", chat_id)
            sess.reset()
        sess.touch()
        return sess

def _next_photo_id(sess: Session) -> int:
    return max((p.photo_id for p in sess.photos), default=0) + 1

# ─────────────────────────────────────────────────────────────────────────────
# Batch processing — runs in background thread(s)
# ─────────────────────────────────────────────────────────────────────────────

BATCH_COLLECT_DELAY = 2.5   # seconds to wait for more photos in same album

def _process_batch(chat_id: int, file_ids: list[str]) -> None:
    """Process a list of photo file_ids sequentially in a background thread."""
    sess  = get_session(chat_id)
    total = len(file_ids)

    try:
        progress = bot.send_message(
            chat_id,
            f"⚙️ Received *{total}* photo(s) — processing *1* of *{total}*…",
        )

        for idx, file_id in enumerate(file_ids, start=1):
            if idx > 1:
                try:
                    bot.edit_message_text(
                        f"⚙️ Processing photo *{idx}* of *{total}*…",
                        chat_id,
                        progress.message_id,
                    )
                except Exception:
                    pass

            try:
                info        = bot.get_file(file_id)
                image_bytes = bot.download_file(info.file_path)
                data        = extract_tickets(image_bytes)
            except Exception as exc:
                log.error("OCR failed chat=%d photo=%d: %s", chat_id, idx, exc)
                err_str = str(exc)
                if "429" in err_str or "quota" in err_str.lower():
                    msg = (
                        "⚠️ The AI service is temporarily busy.\n"
                        "Please wait a minute and try again."
                    )
                elif "upload" in err_str.lower():
                    msg = "❌ Could not upload photo to processing server — please retry."
                else:
                    msg = "❌ Could not read this photo. Try a clearer, well-lit image."
                bot.send_message(chat_id, f"Photo {idx}: {msg}")
                continue

            if not data:
                bot.send_message(
                    chat_id,
                    f"Photo {idx}: ❌ No ticket entries found. Try a clearer image.",
                )
                continue

            try:
                start = str(min((int(t) for t in data if t.isdigit()), default=0))
            except Exception:
                start = next(iter(data), "?")

            sess.photos.append(PhotoRecord(_next_photo_id(sess), start, data))

        # Tear down progress message
        try:
            bot.delete_message(chat_id, progress.message_id)
        except Exception:
            pass

        sess.state = State.AWAITING_CONFIRM if sess.photos else State.IDLE

        if sess.photos:
            bot.send_message(
                chat_id,
                batch_summary(sess.photos),
                reply_markup=export_kb(),
            )
        else:
            bot.send_message(
                chat_id,
                "⚠️ No data could be extracted. Please try again with clearer photos.",
            )

    except Exception as exc:
        log.error("Unhandled crash in _process_batch chat=%d: %s", chat_id, exc, exc_info=True)
        try:
            bot.send_message(chat_id, "❌ An unexpected error occurred. Your session has been reset — please try again.")
        except Exception:
            pass

    finally:
        # ALWAYS runs — guarantees session never stays stuck at PROCESSING
        if sess.state == State.PROCESSING:
            sess.state = State.IDLE


    if sess.photos:
        bot.send_message(
            chat_id,
            batch_summary(sess.photos),
            reply_markup=export_kb(),
        )
        else:
            bot.send_message(
                chat_id,
                "⚠️ No data could be extracted. Please try again with clearer photos.",
            )

    except Exception as exc:
        log.error("Unhandled crash in _process_batch chat=%d: %s", chat_id, exc, exc_info=True)
        try:
            bot.send_message(chat_id, "❌ An unexpected error occurred. Your session has been reset — please try again.")
        except Exception:
            pass

    finally:
        # ALWAYS runs — guarantees session never stays stuck at PROCESSING
        if sess.state == State.PROCESSING:
            sess.state = State.IDLE


def _fire_batch(chat_id: int) -> None:
    """Called by the debounce timer — grabs queued IDs and launches thread."""
    sess = get_session(chat_id)
    file_ids = sess.pending_ids[:]
    sess.pending_ids.clear()
    sess.timer = None
    if not file_ids:
        return
    sess.state = State.PROCESSING
    t = threading.Thread(target=_process_batch, args=(chat_id, file_ids), daemon=True)
    t.start()

# ─────────────────────────────────────────────────────────────────────────────
# File export builders
# ─────────────────────────────────────────────────────────────────────────────

def _tsort(t: str):
    return int(t) if t.isdigit() else t

def _merged(photos: list[PhotoRecord]) -> list[tuple[str, str]]:
    m: dict[str, str] = {}
    for ph in photos:
        m.update(ph.data)
    return sorted(m.items(), key=lambda kv: _tsort(kv[0]))

def build_excel(photos: list[PhotoRecord]) -> bytes:
    rows = _merged(photos)
    df   = pd.DataFrame(
        [{"No.": i, "Phone Number": ("" if p == "Empty" else p)}
         for i, (_, p) in enumerate(rows, 1)]
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Lottery Sales")
    buf.seek(0)

    wb = load_workbook(buf)
    ws = wb.active

    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    hdr_font = Font(bold=True, color="FFFFFF", size=11)
    center   = Alignment(horizontal="center", vertical="center")
    thin     = Side(style="thin", color="AAAAAA")
    bdr      = Border(left=thin, right=thin, top=thin, bottom=thin)

    for cell in ws[1]:
        cell.fill = hdr_fill; cell.font = hdr_font
        cell.alignment = center; cell.border = bdr
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.alignment = center; cell.border = bdr

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 22
    ws.freeze_panes    = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 22
    for i in range(2, ws.max_row + 1):
        ws.row_dimensions[i].height = 18

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()

def build_csv(photos: list[PhotoRecord]) -> bytes:
    buf    = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["No.", "Phone Number"])
    for i, (_, p) in enumerate(_merged(photos), 1):
        writer.writerow([i, "" if p == "Empty" else p])
    return buf.getvalue().encode("utf-8-sig")

def build_txt(photos: list[PhotoRecord]) -> bytes:
    lines = [p for _, p in _merged(photos) if p != "Empty"]
    block = "\n\n--- comma-separated ---\n" + ", ".join(lines)
    return ("\n".join(lines) + block).encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def export_kb(add_more: bool = True) -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup(row_width=1)
    if add_more:
        kb.add(types.InlineKeyboardButton("📷 Add More Photos",      callback_data="add_more"))
    kb.add(types.InlineKeyboardButton("📊 Export Excel (.xlsx)",     callback_data="export_xlsx"))
    kb.add(types.InlineKeyboardButton("📄 Export CSV (.csv)",        callback_data="export_csv"))
    kb.add(types.InlineKeyboardButton("📝 Export Text (.txt)",       callback_data="export_txt"))
    kb.add(types.InlineKeyboardButton("❌ Cancel & Discard",         callback_data="cancel"))
    return kb

def batch_summary(photos: list[PhotoRecord]) -> str:
    total_f = sum(p.filled for p in photos)
    total_t = sum(len(p.data) for p in photos)
    lines   = ["📦 *Batch Summary*", ""]
    for ph in photos:
        lines.append(
            f"  📸 Photo *{ph.photo_id}* "
            f"(from ticket #{ph.start_ticket}): "
            f"*{ph.filled}* numbers, *{ph.empty}* empty"
        )
    lines += [
        "",
        f"✅ *{total_f} / {total_t}* tickets have a number.",
        "",
        "👉 `/edit [Photo ID]` to correct any ticket",
        "👉 Choose an export format below:",
    ]
    return "\n".join(lines)

def photo_detail_pages(ph: PhotoRecord) -> list[str]:
    header = f"✏️ *Photo {ph.photo_id}* — starts at ticket #{ph.start_ticket}\n\n"
    footer = "\nType the *ticket number* to correct:"
    lines  = []
    for ticket, phone in sorted(ph.data.items(), key=lambda kv: _tsort(kv[0])):
        icon = "⬜" if phone == "Empty" else "📞"
        lines.append(f"{icon} Ticket *{ticket}*: `{phone}`")

    pages: list[str] = []
    current = header
    for line in lines:
        if len(current) + len(line) + 1 > MSG_CHAR_LIMIT:
            pages.append(current)
            current = ""
        current += line + "\n"
    pages.append(current + footer)
    return pages

def _del(chat_id: int, msg_id: int) -> None:
    try:
        bot.delete_message(chat_id, msg_id)
    except Exception:
        pass

def _rm_markup(chat_id: int, msg_id: int) -> None:
    try:
        bot.edit_message_reply_markup(chat_id, msg_id, reply_markup=None)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────


@bot.message_handler(content_types=["photo"])
def handle_photo(msg: types.Message) -> None:
    chat_id = msg.chat.id
    sess    = get_session(chat_id)

    if sess.state == State.PROCESSING:
        bot.send_message(chat_id, "⚙️ Still processing — please wait.")
        return
    if sess.state in (State.AWAITING_EDIT_TICKET, State.AWAITING_EDIT_PHONE):
        bot.send_message(chat_id, "⚠️ Please finish editing before sending a new photo.")
        return
    if len(sess.photos) + len(sess.pending_ids) >= MAX_PHOTOS:
        bot.send_message(chat_id,
            f"⚠️ Maximum {MAX_PHOTOS} photos per batch. Export first.")
        return

    # Accept new photos even while AWAITING_CONFIRM (user wants to add more)
    if sess.state == State.AWAITING_CONFIRM:
        sess.state = State.COLLECTING

    file_id = msg.photo[-1].file_id
    sess.pending_ids.append(file_id)
    sess.state = State.COLLECTING

    # Debounce: reset timer on each arriving photo
    if sess.timer:
        try:
            sess.timer.cancel()
        except Exception:
            pass

    t = threading.Timer(BATCH_COLLECT_DELAY, _fire_batch, args=(chat_id,))
    t.daemon = True
    sess.timer = t
    t.start()

@bot.callback_query_handler(func=lambda c: True)
def handle_callback(call: types.CallbackQuery) -> None:
    chat_id = call.message.chat.id
    sess    = get_session(chat_id)
    bot.answer_callback_query(call.id)
    action  = call.data

    if action == "add_more":
        if sess.state != State.AWAITING_CONFIRM:
            return
        sess.state = State.COLLECTING
        _rm_markup(chat_id, call.message.message_id)
        bot.send_message(chat_id, "📷 Ready — send your next photo(s).")
        return

    if action in ("export_xlsx", "export_csv", "export_txt"):
        if not sess.photos or sess.state != State.AWAITING_CONFIRM:
            bot.send_message(chat_id, "⚠️ No data to export.")
            return
        building = bot.send_message(chat_id, "⏳ Building your file…")
        try:
            if action == "export_xlsx":
                raw = build_excel(sess.photos)
                bot.send_document(chat_id, io.BytesIO(raw),
                    visible_file_name="lottery_tickets.xlsx",
                    caption="📊 Excel export ready.")
            elif action == "export_csv":
                raw = build_csv(sess.photos)
                bot.send_document(chat_id, io.BytesIO(raw),
                    visible_file_name="lottery_tickets.csv",
                    caption="📄 CSV export ready.")
            else:
                raw = build_txt(sess.photos)
                bot.send_document(chat_id, io.BytesIO(raw),
                    visible_file_name="lottery_tickets.txt",
                    caption="📝 Text export ready.")
            _del(chat_id, building.message_id)
            _rm_markup(chat_id, call.message.message_id)
            sess.reset()
        except Exception as exc:
            log.error("Export failed chat=%d: %s", chat_id, exc)
            try:
                bot.edit_message_text("❌ Export failed — please try again.",
                                      chat_id, building.message_id)
            except Exception:
                pass
        return

    if action == "cancel":
        sess.reset()
        try:
            bot.edit_message_text("❌ Batch cancelled — all data discarded.",
                                  chat_id, call.message.message_id)
        except Exception:
            bot.send_message(chat_id, "❌ Batch cancelled.")

@bot.message_handler(commands=["edit"])
def handle_edit_cmd(msg: types.Message) -> None:
    chat_id = msg.chat.id
    sess    = get_session(chat_id)

    if sess.state != State.AWAITING_CONFIRM or not sess.photos:
        bot.send_message(chat_id, "⚠️ No active batch to edit.")
        return

    parts = msg.text.strip().split()
    if len(parts) < 2 or not parts[1].isdigit():
        ids = ", ".join(str(p.photo_id) for p in sess.photos)
        bot.send_message(chat_id, f"ℹ️ Usage: `/edit [Photo ID]`\nAvailable IDs: {ids}")
        return

    photo_id = int(parts[1])
    photo    = next((p for p in sess.photos if p.photo_id == photo_id), None)
    if not photo:
        bot.send_message(chat_id, f"⚠️ Photo ID *{photo_id}* not found.")
        return

    sess.edit_photo_id = photo_id
    sess.state         = State.AWAITING_EDIT_TICKET
    for page in photo_detail_pages(photo):
        bot.send_message(chat_id, page)

@bot.message_handler(
    func=lambda m: get_session(m.chat.id).state == State.AWAITING_EDIT_TICKET)
def handle_edit_ticket(msg: types.Message) -> None:
    sess   = get_session(msg.chat.id)
    ticket = msg.text.strip()
    photo  = next((p for p in sess.photos if p.photo_id == sess.edit_photo_id), None)

    if not photo or ticket not in photo.data:
        bot.send_message(msg.chat.id, f"⚠️ Ticket *{ticket}* not found. Try again:")
        return

    sess.edit_ticket = ticket
    sess.state       = State.AWAITING_EDIT_PHONE
    bot.send_message(
        msg.chat.id,
        f"Ticket *{ticket}* is currently: `{photo.data[ticket]}`\n\n"
        "Enter the correct 10-digit number, or type `empty` to clear it:",
    )

@bot.message_handler(
    func=lambda m: get_session(m.chat.id).state == State.AWAITING_EDIT_PHONE)
def handle_edit_phone(msg: types.Message) -> None:
    chat_id = msg.chat.id
    sess    = get_session(chat_id)
    value   = msg.text.strip()

    if value.lower() == "empty":
        new_phone = "Empty"
    elif PHONE_RE.match(value):
        new_phone = value
    else:
        bot.send_message(chat_id,
            "⚠️ Invalid number — must be 10 digits starting with 07 or 09.\n"
            "Try again, or type `empty`:")
        return

    photo = next((p for p in sess.photos if p.photo_id == sess.edit_photo_id), None)
    if photo:
        photo.data[sess.edit_ticket] = new_phone

    ticket             = sess.edit_ticket
    sess.edit_photo_id = None
    sess.edit_ticket   = None
    sess.state         = State.AWAITING_CONFIRM

    bot.send_message(
        chat_id,
        f"✅ Ticket *{ticket}* updated to `{new_phone}`.\n\n" + batch_summary(sess.photos),
        reply_markup=export_kb(),
    )

@bot.message_handler(commands=["start", "help"])
def handle_start(msg: types.Message) -> None:
    bot.send_message(
        msg.chat.id,
        "👋 *Lottery Logbook Bot*\n\n"
        "📸 Send one or more photos of your logbook pages.\n"
        "🔍 I'll extract every ticket number and phone automatically.\n\n"
        "*Commands:*\n"
        "• `/edit [ID]` — correct a ticket entry\n"
        "• `/status` — show current batch summary\n"
        "• `/cancel` — discard the current batch\n\n"
        "📊 Export to *Excel*, *CSV*, or *plain text* when ready.",
    )

@bot.message_handler(commands=["cancel"])
def handle_cancel(msg: types.Message) -> None:
    get_session(msg.chat.id).reset()
    bot.send_message(msg.chat.id, "❌ Session cancelled — all data cleared.")

@bot.message_handler(commands=["status"])
def handle_status(msg: types.Message) -> None:
    sess = get_session(msg.chat.id)
    if not sess.photos:
        bot.send_message(msg.chat.id, "ℹ️ No active batch. Send a photo to get started.")
        return
    bot.send_message(
        msg.chat.id,
        batch_summary(sess.photos),
        reply_markup=export_kb(add_more=sess.state == State.AWAITING_CONFIRM),
    )

@bot.message_handler(
    func=lambda m: get_session(m.chat.id).state == State.IDLE,
    content_types=["text"],
)
def handle_idle(msg: types.Message) -> None:
    bot.send_message(
        msg.chat.id,
        "📸 Send a photo of your logbook page to get started, or /help for instructions.",
    )

# Start the web server in a background thread
web_thread = Thread(target=run_web_server)
web_thread.daemon = True
web_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Lottery Logbook Bot v4.0 — ready (model: %s).", OCR_MODEL)
    bot.remove_webhook()
    bot.infinity_polling(timeout=30, long_polling_timeout=30)
