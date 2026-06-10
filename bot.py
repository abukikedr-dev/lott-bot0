"""
Lottery Logbook Bot — v4.1 (Native Multi-Image Batching + Key Rotation)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import time
import json
import logging
import itertools
import threading
from typing import Optional, Dict, List
from threading import Thread

import telebot
from telebot import types
import requests
import google.generativeai as genai

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Environment & Keys ───────────────────────────────────────────────────────
TELEGRAM_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_TOKEN")

# Parse multiple keys from env (comma-separated). Fallback to single key if needed.
keys_env = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
GEMINI_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]

if not GEMINI_KEYS:
    raise ValueError("No Gemini API keys found. Please set GEMINI_API_KEYS in your environment variables.")

_key_cycle = itertools.cycle(GEMINI_KEYS)
_KEY_LOCK = threading.Lock()

def get_next_key() -> str:
    """Safely grabs the next API key in the rotation."""
    with _KEY_LOCK:
        return next(_key_cycle)

PROXY_URL: Optional[str] = None
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="Markdown")

# ── Constants ────────────────────────────────────────────────────────────────
OCR_MODEL       = "gemini-3.5-flash"
MAX_RETRIES     = 6
BACKOFF_BASE    = 1.5
BACKOFF_CAP     = 64.0
SESSION_TTL     = 3600

PHONE_RE = re.compile(r"^0[79]\d{8}$")
FILES_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files?uploadType=multipart"

# ── State Management ─────────────────────────────────────────────────────────
class State:
    IDLE = "IDLE"
    PROCESSING = "PROCESSING"
    AWAITING_CONFIRM = "AWAITING_CONFIRM"

class PhotoRecord:
    def __init__(self, photo_id: int, start_ticket: str, data: dict):
        self.photo_id = photo_id
        self.start_ticket = start_ticket
        self.data = data

class Session:
    def __init__(self):
        self.state = State.IDLE
        self.photos: List[PhotoRecord] = []
        self.last_active = time.time()

    def reset(self):
        self.state = State.IDLE
        self.photos.clear()
        self.last_active = time.time()

_sessions: Dict[int, Session] = {}

def get_session(chat_id: int) -> Session:
    if chat_id not in _sessions:
        _sessions[chat_id] = Session()
    _sessions[chat_id].last_active = time.time()
    return _sessions[chat_id]

def _next_photo_id(sess: Session) -> int:
    return max([p.photo_id for p in sess.photos] + [0]) + 1

# ── Prompt & Parser ──────────────────────────────────────────────────────────
_PROMPT = """
You are a specialist OCR assistant for Ethiopian handwritten lottery logbooks.
You will receive multiple photos of logbook pages.

PAGE LAYOUT — each page has one or more vertical column-pairs:
  LEFT  : sequential ticket numbers (handwritten integers, ascending).
  RIGHT : customer name in Amharic (IGNORE) and a 10-digit Ethiopian mobile number starting with 09 or 07.

SPANNING RULE:
  A phone number written large may visually span several ticket rows.
  Assign that SAME number to EVERY ticket it covers. Never mark a covered row "Empty".

VALIDATION:
  • Phone must be exactly 10 digits, starting with 07 or 09.
  • If unreadable or absent → "Empty". Never guess or invent digits.

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown. No prose.
Group the output by the image labels provided in the prompt (e.g., image_1, image_2).
{"image_1": [{"ticket": "1", "phone": "0916039018"}], "image_2": [{"ticket": "2", "phone": "Empty"}]}
"""

def _parse_ocr(raw: str) -> dict[str, dict[str, str]]:
    raw = re.sub(r"^\s*```[a-zA-Z]*\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {raw[:200]}")
    
    parsed = json.loads(match.group(0))
    result = {}
    
    for img_key, tickets_list in parsed.items():
        if not isinstance(tickets_list, list):
            continue
            
        img_result = {}
        for item in tickets_list:
            if not isinstance(item, dict): continue
            ticket = str(item.get("ticket", "")).strip()
            phone  = str(item.get("phone",  "Empty")).strip()
            if not ticket: continue
            if phone != "Empty" and not PHONE_RE.match(phone):
                phone = "Empty"
            img_result[ticket] = phone
            
        result[img_key] = img_result
        
    return result

# ── Files API Helpers ────────────────────────────────────────────────────────
def _upload_image(image_bytes: bytes, api_key: str, mime_type: str = "image/jpeg") -> str:
    proxies = {"https": PROXY_URL, "http": PROXY_URL} if PROXY_URL else None
    metadata_part = json.dumps({"file": {"display_name": "logbook_page"}})
    boundary = "-----lottery_bot_boundary"
    
    body = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n\r\n"
        f"{metadata_part}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--".encode()

    headers = {
        "X-Goog-Api-Key": api_key,
        "Content-Type": f"multipart/related; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    resp = requests.post(FILES_UPLOAD_URL, headers=headers, data=body, proxies=proxies, timeout=60)
    resp.raise_for_status()
    file_uri = resp.json()["file"]["uri"]
    log.info("Uploaded image → %s (Using key: ...%s)", file_uri, api_key[-4:])
    return file_uri

def _delete_file(file_uri: str, api_key: str) -> None:
    try:
        proxies = {"https": PROXY_URL, "http": PROXY_URL} if PROXY_URL else None
        name = file_uri.split("/v1beta/")[-1] if "/v1beta/" in file_uri else file_uri
        url = f"https://generativelanguage.googleapis.com/v1beta/{name}"
        requests.delete(url, headers={"X-Goog-Api-Key": api_key}, proxies=proxies, timeout=10)
        log.info("Deleted remote file %s", name)
    except Exception as e:
        log.debug("File delete failed (non-fatal): %s", e)

# ── Core AI Extractor ────────────────────────────────────────────────────────
def extract_tickets(images_bytes: list[bytes]) -> dict[str, dict[str, str]]:
    uploaded_uris = []
    current_key = get_next_key()
    
    try:
        for image_bytes in images_bytes:
            mime_type = "image/jpeg"
            if image_bytes.startswith(b"\x89PNG"): mime_type = "image/png"
            elif image_bytes.startswith(b"\xff\xd8\xff"): mime_type = "image/jpeg"
            elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]: mime_type = "image/webp"
            elif image_bytes.startswith(b"GIF8"): mime_type = "image/gif"
            
            for attempt in range(3):
                try:
                    file_uri = _upload_image(image_bytes, current_key, mime_type=mime_type)
                    uploaded_uris.append((file_uri, mime_type))
                    break
                except Exception as e:
                    if attempt == 2: raise
                    log.warning("Upload attempt %d failed: %s — retrying", attempt + 1, e)
                    time.sleep(2 ** attempt)

        with _KEY_LOCK:
            genai.configure(api_key=current_key)
            
            for file_uri, _ in uploaded_uris:
                for _ in range(10):
                    file_info = genai.get_file(file_uri.split("/")[-1])
                    if file_info.state.name == "ACTIVE": break
                    time.sleep(1)

            model = genai.GenerativeModel(OCR_MODEL)

            prompt_parts = [_PROMPT]
            for idx, (file_uri, mime_type) in enumerate(uploaded_uris, start=1):
                prompt_parts.append(f"image_{idx}:")
                prompt_parts.append({"file_data": {"mime_type": mime_type, "file_uri": file_uri}})

            sleep_time = BACKOFF_BASE
            last_exc = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = model.generate_content(
                        prompt_parts,
                        generation_config={"temperature": 0.1, "max_output_tokens": 8192},
                    )
                    return _parse_ocr(response.text)
                except (json.JSONDecodeError, ValueError) as e:
                    last_exc = e
                    if attempt >= 2: raise RuntimeError(f"Bad JSON twice: {e}") from e
                    continue
                except Exception as e:
                    last_exc = e
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower() or "503" in err_str:
                        m = re.search(r"retry[_ ]delay.*?(\d+\.?\d*)", err_str, re.I)
                        actual_sleep = min(float(m.group(1)) + 1.0 if m else sleep_time, BACKOFF_CAP)
                        log.warning("Rate-limit/503 on attempt %d — sleeping %.1fs", attempt, actual_sleep)
                        time.sleep(actual_sleep)
                        sleep_time = min(sleep_time * 2, BACKOFF_CAP)
                        continue
                    raise

            raise RuntimeError(f"OCR failed after {MAX_RETRIES} attempts. Last: {last_exc}") from last_exc

    finally:
        for file_uri, _ in uploaded_uris:
            _delete_file(file_uri, current_key)

# ── Batch Processor ──────────────────────────────────────────────────────────
def batch_summary(photos: List[PhotoRecord]) -> str:
    total_tickets = sum(len(p.data) for p in photos)
    return f"📊 *Batch Summary*\nPhotos processed: {len(photos)}\nTotal tickets: {total_tickets}"

def export_kb(add_more: bool = True) -> types.InlineKeyboardMarkup:
    kb = types.InlineKeyboardMarkup()
    if add_more:
        kb.add(types.InlineKeyboardButton("➕ Add More Photos", callback_data="add_more"))
    kb.add(types.InlineKeyboardButton("📥 Export CSV", callback_data="export_csv"))
    kb.add(types.InlineKeyboardButton("❌ Cancel", callback_data="cancel"))
    return kb

def _process_batch(chat_id: int, file_ids: list[str]) -> None:
    sess = get_session(chat_id)
    total = len(file_ids)

    try:
        progress = bot.send_message(chat_id, f"⚙️ Downloading *{total}* photo(s) to process as a single batch…")

        image_bytes_list = []
        for file_id in file_ids:
            info = bot.get_file(file_id)
            image_bytes_list.append(bot.download_file(info.file_path))

        bot.edit_message_text(f"⚙️ Running AI OCR on all *{total}* photos at once (saving quota)…", chat_id, progress.message_id)

        try:
            batch_data = extract_tickets(image_bytes_list)
        except Exception as exc:
            log.error("OCR failed batch for chat=%d: %s", chat_id, exc)
            err_str = str(exc)
            if "429" in err_str or "quota" in err_str.lower(): 
                msg = "⚠️ The AI service is temporarily busy. Please try again."
            else: 
                msg = "❌ Could not process this batch. Try clearer images."
            bot.send_message(chat_id, msg)
            return

        for idx in range(1, total + 1):
            data = batch_data.get(f"image_{idx}")
            if not data and len(batch_data) >= idx:
                data = list(batch_data.values())[idx - 1]

            if not data:
                bot.send_message(chat_id, f"Photo {idx}: ❌ No ticket entries found.")
                continue

            try: start = str(min((int(t) for t in data if t.isdigit()), default=0))
            except Exception: start = next(iter(data), "?")

            sess.photos.append(PhotoRecord(_next_photo_id(sess), start, data))

        try: bot.delete_message(chat_id, progress.message_id)
        except Exception: pass

        sess.state = State.AWAITING_CONFIRM if sess.photos else State.IDLE

        if sess.photos: 
            bot.send_message(chat_id, batch_summary(sess.photos), reply_markup=export_kb())
        else: 
            bot.send_message(chat_id, "⚠️ No data could be extracted. Please try again with clearer photos.")

    except Exception as exc:
        log.error("Unhandled crash in _process_batch chat=%d: %s", chat_id, exc, exc_info=True)
        try: bot.send_message(chat_id, "❌ An unexpected error occurred. Your session has been reset.")
        except Exception: pass
    finally:
        if sess.state == State.PROCESSING: sess.state = State.IDLE

# ── Handlers ─────────────────────────────────────────────────────────────────
@bot.message_handler(content_types=['photo'])
def handle_photos(msg: types.Message):
    sess = get_session(msg.chat.id)
    if sess.state == State.PROCESSING:
        bot.reply_to(msg, "⏳ Please wait, I am currently processing a batch...")
        return
        
    sess.state = State.PROCESSING
    # If using media groups, Telegram sends photos individually. You might need 
    # external media group handlers depending on your bot's specific architecture.
    # For a single photo test:
    file_id = msg.photo[-1].file_id
    
    Thread(target=_process_batch, args=(msg.chat.id, [file_id]), daemon=True).start()

@bot.message_handler(commands=["start", "help"])
def handle_start(msg: types.Message) -> None:
    bot.send_message(
        msg.chat.id,
        "👋 Welcome! Send me photos of your logbook to extract tickets.\n\n"
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

# ── Polling Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Lottery Logbook Bot (Multi-Image Batching & Key Rotation)...")
    while True:
        try:
            bot.polling(none_stop=True, timeout=60)
        except Exception as e:
            log.error("Bot polling failed: %s. Restarting in 5s...", e)
            time.sleep(5)
