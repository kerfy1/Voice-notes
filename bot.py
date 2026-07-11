"""
Telegram-бот: голосовое → Whisper → Claude (OpenRouter) → карточка в Notion.

Поток:
  1. Пользователь шлёт voice / audio / forwarded audio (в т.ч. подкасты 30+ мин).
  2. Файл скачивается, конвертируется в mono mp3 через ffmpeg;
     длинные аудио режутся на чанки по CHUNK_SECONDS (по умолчанию 5 мин).
  3. Каждый чанк уходит в Whisper на OpenRouter (endpoint /v1/audio/transcriptions,
     OpenAI-совместимый), с прогресс-апдейтами в чат.
  4. Бот показывает превью транскрипции и спрашивает про акценты.
  5. Если акценты — ждёт текст.
  6. Транскрипция (+ акценты) уходят в Claude через OpenRouter,
     возвращается JSON: {title, summary, tag}.
  7. Создаётся страница в Notion-базе с колонками: Назва ідеї / Суть / Дата / Тег;
     длинный контент дописывается батчами по 90 блоков (лимит Notion — 100).
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# ---------- config ----------
# Ищем файл с ключами рядом со скриптом. Принимаем несколько вариантов имени,
# т.к. Windows любит незаметно добавлять .env или .txt.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_CANDIDATES = [
    _SCRIPT_DIR / "api_key",
    _SCRIPT_DIR / "api_key.env",
    _SCRIPT_DIR / ".env",
]
_ENV_PATH = next((p for p in _ENV_CANDIDATES if p.is_file()), None)
if _ENV_PATH is not None:
    load_dotenv(_ENV_PATH)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# Проверяем обязательные ключи ДО инициализации клиентов, чтобы получить
# понятное сообщение, а не traceback из недр openai/notion SDK.
_missing = [
    name for name, value in {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
        "NOTION_TOKEN": NOTION_TOKEN,
        "NOTION_DATABASE_ID": NOTION_DATABASE_ID,
    }.items() if not value
]
if _missing:
    if _ENV_PATH is None:
        raise SystemExit(
            "❌ Не найден файл с ключами.\n"
            f"   Ищу здесь: {', '.join(str(p) for p in _ENV_CANDIDATES)}\n"
            "   Создай файл `api_key` (без расширения) рядом с bot.py."
        )
    raise SystemExit(
        f"❌ В файле с ключами не хватает: {', '.join(_missing)}\n"
        f"   Загружено из: {_ENV_PATH}\n"
        f"   Проверь, что файл содержит строки вида `KEY=значение` без пробелов и кавычек,\n"
        f"   и сохранён в кодировке UTF-8."
    )

# Модели через OpenRouter (один аккаунт на всё)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-1")

# Потолок токенов ответа Claude. Обязательно задавать явно: иначе OpenRouter
# резервирует кредит под всё окно вывода модели и при малом остатке даёт 402.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4000"))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- long audio handling ---
# whisper-1 обрабатывает ~5× реального времени, а у OpenRouter таймаут ~60 сек
# на аудио-запрос → безопасный размер чанка ~5 минут.
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "300"))
# Если чанк получается меньше этого — не режем, гоним одним куском.
NO_SPLIT_UNDER = int(os.getenv("NO_SPLIT_UNDER", "420"))  # 7 мин
# Предохранитель от случайно пересланного 8-часового аудиокурса.
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "7200"))  # 2 ч
# Бинарники ffmpeg (можно указать полный путь в .env, если не в PATH).
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# Имена свойств в Notion — вынесены в env, чтобы можно было поменять
# без правки кода (у тебя они на украинском).
PROP_TITLE = os.getenv("NOTION_PROP_TITLE", "Назва ідеї")
PROP_ESSENCE = os.getenv("NOTION_PROP_ESSENCE", "Суть")
PROP_DATE = os.getenv("NOTION_PROP_DATE", "Дата")
PROP_TAG = os.getenv("NOTION_PROP_TAG", "Тег")

# Ограничение доступа: если задан ALLOWED_USER_IDS (через запятую) — пускаем только их.
_allowed = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWED_USER_IDS = {int(x) for x in _allowed.split(",") if x.strip()} if _allowed else None

# Язык транскрипции по умолчанию для кнопки-подсказки.
# Пусто → бот всегда спрашивает. Можно задать "ru"/"uk"/"auto", чтобы пропускать вопрос.
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "").strip().lower()

# Директория для временных аудио (кросс-платформенно: %TEMP% на Windows, /tmp на *nix)
TMP_DIR = Path(tempfile.gettempdir()) / "voice_bot"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ---------- states ----------
CHOOSING_LANGUAGE = 0
WAITING_FOR_HIGHLIGHTS = 1

# ---------- logging ----------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voicebot")

# ---------- clients ----------
# OpenRouter поддерживает OpenAI-совместимый endpoint /v1/audio/transcriptions,
# поэтому используем openai SDK, просто подменив base_url и ключ.
whisper_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    timeout=120.0,
)
notion = NotionClient(auth=NOTION_TOKEN)


# =====================================================================
# helpers
# =====================================================================
def _is_allowed(update: Update) -> bool:
    if ALLOWED_USER_IDS is None:
        return True
    user = update.effective_user
    return bool(user and user.id in ALLOWED_USER_IDS)


async def _reject(update: Update) -> None:
    if update.message:
        await update.message.reply_text("⛔ Доступ запрещён.")


def _run_ffmpeg(cmd: list[str]) -> str:
    """Синхронно запускает ffmpeg/ffprobe, кидает RuntimeError с stderr при ошибке."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]}: {r.stderr.strip() or 'unknown error'}")
    return r.stdout


def _ensure_ffmpeg() -> None:
    if not shutil.which(FFMPEG_BIN) or not shutil.which(FFPROBE_BIN):
        raise RuntimeError(
            "ffmpeg/ffprobe не найдены в PATH. Установи: "
            "`apt install ffmpeg` (Linux) / `brew install ffmpeg` (macOS) / "
            "https://ffmpeg.org/download.html (Windows). "
            "Либо укажи путь в .env через FFMPEG_BIN и FFPROBE_BIN."
        )


def _probe_duration(path: Path) -> float:
    out = _run_ffmpeg([
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(out.strip())


def _to_mp3(src: Path, dst: Path) -> None:
    """
    Пережимаем в mono mp3 64 kbps. Для речи качества более чем достаточно,
    зато размер маленький (30 мин ≈ 14 МБ) — не упрёмся в лимит Whisper 25 МБ.
    """
    _run_ffmpeg([
        FFMPEG_BIN, "-y", "-i", str(src),
        "-ac", "1",       # mono
        "-b:a", "64k",    # 64 kbps
        "-vn",            # никакого видео (для video_note / mp4)
        str(dst),
    ])


def _split_mp3(src: Path, out_dir: Path, chunk_seconds: int) -> list[Path]:
    """Режем mp3 на куски фиксированной длительности."""
    pattern = str(out_dir / "chunk_%03d.mp3")
    _run_ffmpeg([
        FFMPEG_BIN, "-y", "-i", str(src),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern,
    ])
    return sorted(out_dir.glob("chunk_*.mp3"))


def _prepare_audio_sync(voice_path: Path, work_dir: Path) -> tuple[list[Path], float]:
    """
    Готовит аудио к отправке в Whisper.
      → всегда конвертирует в mp3 (унифицированный формат + компрессия).
      → если длинное, режет на чанки CHUNK_SECONDS.
    Возвращает (список путей, общая длительность в секундах).
    """
    _ensure_ffmpeg()
    mp3_path = work_dir / "audio.mp3"
    _to_mp3(voice_path, mp3_path)
    duration = _probe_duration(mp3_path)

    if duration > MAX_AUDIO_SECONDS:
        raise RuntimeError(
            f"Аудио {duration/60:.1f} мин — превышен лимит "
            f"{MAX_AUDIO_SECONDS/60:.0f} мин. Можно поднять MAX_AUDIO_SECONDS в .env."
        )

    if duration <= NO_SPLIT_UNDER:
        return [mp3_path], duration

    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    chunks = _split_mp3(mp3_path, chunks_dir, CHUNK_SECONDS)
    if not chunks:
        # На всякий случай fallback: если сегментация не сработала — гоним целиком.
        return [mp3_path], duration
    return chunks, duration


async def _prepare_audio(voice_path: Path, work_dir: Path) -> tuple[list[Path], float]:
    """Асинхронная обёртка над блокирующим ffmpeg — чтобы не вешать event loop."""
    return await asyncio.to_thread(_prepare_audio_sync, voice_path, work_dir)


# Подсказки-«праймеры» для Whisper: короткий текст на нужном языке помогает
# модели не срываться в перевод и не путать язык (ru ↔ be).
_LANG_PROMPTS = {
    "ru": "Это транскрипция голосового сообщения на русском языке.",
    "uk": "Це транскрипція голосового повідомлення українською мовою.",
}


async def _transcribe_chunk(audio_path: Path, language: str | None) -> str:
    """
    Одно обращение к Whisper через OpenRouter.
    language: "ru" / "uk" / None (auto). Если задан — форсим язык и подсказку,
    что убирает срыв в английский перевод и путаницу с белорусским.
    """
    kwargs = {
        "model": WHISPER_MODEL,
        "extra_headers": {
            "HTTP-Referer": "https://github.com/local/voice-notion-bot",
            "X-Title": "Voice Notion Bot",
        },
    }
    if language:
        kwargs["language"] = language          # ISO-639-1: "ru" / "uk"
        prompt = _LANG_PROMPTS.get(language)
        if prompt:
            kwargs["prompt"] = prompt

    with audio_path.open("rb") as f:
        result = await whisper_client.audio.transcriptions.create(file=f, **kwargs)
    return (result.text or "").strip()


async def transcribe(voice_path: Path, status_msg, language: str | None = None) -> str:
    """
    Полный цикл транскрипции с учётом длинных аудио.
    status_msg — Telegram-сообщение, в которое пишем прогресс.
    language — "ru"/"uk"/None(auto), передаётся в Whisper для точности.
    """
    work_dir = Path(tempfile.mkdtemp(prefix="voice_", dir=str(TMP_DIR)))
    try:
        await status_msg.edit_text("🔧 Готовлю аудио (ffmpeg)…")
        chunks, duration = await _prepare_audio(voice_path, work_dir)
        mins = duration / 60

        # Короткое — один запрос
        if len(chunks) == 1:
            await status_msg.edit_text(f"📝 Расшифровываю ({mins:.1f} мин)…")
            return await _transcribe_chunk(chunks[0], language)

        # Длинное — по чанкам, последовательно, с прогрессом
        total = len(chunks)
        log.info("Long audio: %.1f min → %d chunks × %ds (lang=%s)",
                 mins, total, CHUNK_SECONDS, language or "auto")
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            await status_msg.edit_text(
                f"📝 Расшифровываю: часть {i}/{total} "
                f"(всего ~{mins:.0f} мин)…"
            )
            try:
                text = await _transcribe_chunk(chunk, language)
            except Exception as e:  # noqa: BLE001
                # Один сбойный чанк не должен убивать всё — помечаем и едем дальше
                log.exception("Chunk %d/%d failed", i, total)
                text = f"[⚠️ часть {i}/{total} не расшифрована: {e}]"
            parts.append(text)

        return "\n\n".join(p for p in parts if p)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _strip_markdown(text: str) -> str:
    """
    Подчищает markdown-разметку из текста модели — на случай, если Claude
    всё-таки вставил её вопреки инструкции. Telegram (в HTML-режиме) и Notion
    показывают её как сырой текст, поэтому убираем.
    """
    if not text:
        return text
    # **жирный** и __жирный__ → просто текст
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # *курсив* → текст (но не трогаем маркеры списка "* " в начале строки)
    text = re.sub(r"(?<!\n)(?<!^)\*(?!\s)(.+?)\*", r"\1", text)
    # `код` → текст
    text = re.sub(r"`(.+?)`", r"\1", text)
    # заголовки "# ", "## " в начале строки → убираем решётки
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    # маркеры списка "* " или "+ " в начале строки → унифицируем в "- "
    text = re.sub(r"^(\s*)[*+]\s+", r"\1- ", text, flags=re.MULTILINE)
    return text


async def analyze_with_claude(transcription: str, highlights: str | None) -> dict:
    """Отправляет транскрипцию (+ акценты) в Claude через OpenRouter, ждёт JSON."""
    system_prompt = (
        "Ты помощник, который делает структурированные конспекты из "
        "расшифровок голосовых сообщений на русском или украинском языке.\n\n"
        "Проанализируй текст и верни строго валидный JSON без markdown-обрамления "
        "и без лишних комментариев, с полями:\n"
        "  - title: короткий заголовок конспекта (до 60 символов), "
        "на том же языке, что и оригинал.\n"
        "  - summary: структурированный конспект. Выдели ключевые тезисы "
        "маркированным списком (используй `- ` для пунктов), при необходимости — "
        "подсекции. Пиши по делу, без воды. Тот же язык, что и оригинал.\n"
        "  - tag: одно слово-тег для категоризации, с большой буквы. "
        "Примеры: Ідея, Подкаст, Роздуми, План, Навчання, Робота, Цитата, Задача. "
        "На том же языке, что и оригинал.\n\n"
        "ВАЖНО про форматирование summary: НЕ используй markdown-разметку "
        "внутри текста — никаких звёздочек ** для жирного, никаких # для "
        "заголовков, никаких backtick'ов. Только обычный текст. "
        "Для списков используй дефис `- ` в начале строки. "
        "Для подзаголовков — просто строка текста с двоеточием в конце "
        "(например, «Основные тезисы:»), без всякой разметки.\n\n"
        "Если пользователь прислал акценты — обязательно отрази их в summary "
        "(вынеси в отдельный блок «Акценти» / «Акценты» в начале)."
    )

    user_content = f"Транскрипция:\n\"\"\"\n{transcription}\n\"\"\""
    if highlights:
        user_content += (
            f"\n\nАкценты от пользователя (выделить отдельно):\n"
            f"\"\"\"\n{highlights}\n\"\"\""
        )

    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                # Опциональные заголовки для аналитики OpenRouter:
                "HTTP-Referer": "https://github.com/local/voice-notion-bot",
                "X-Title": "Voice Notion Bot",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                # ВАЖНО: без явного max_tokens OpenRouter резервирует под ответ
                # стоимость по всему окну вывода модели, что при низком остатке
                # на ключе даёт 402 "requires more credits, or fewer max_tokens".
                # Конспекта с запасом хватает 4000 токенов.
                "max_tokens": MAX_OUTPUT_TOKENS,
            },
        )
        # Понятные сообщения для частых ошибок OpenRouter вместо сырого traceback.
        if r.status_code == 402:
            # 402 бывает не только при нулевом балансе: OpenRouter резервирует
            # под ответ стоимость по max_tokens. При низком остатке на КЛЮЧЕ
            # (limit на ключе, а не общий баланс) — тоже 402.
            detail = ""
            try:
                detail = r.json().get("error", {}).get("message", "")
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                "OpenRouter вернул 402 (оплата/лимиты). Проверь:\n"
                "• остаток по КЛЮЧУ (не только общий баланс): "
                "https://openrouter.ai/settings/keys — подними limit ключа;\n"
                "• общий баланс: https://openrouter.ai/settings/credits;\n"
                "• уменьши MAX_OUTPUT_TOKENS в файле api_key."
                + (f"\nОтвет сервера: {detail}" if detail else "")
            )
        if r.status_code == 401:
            raise RuntimeError(
                "неверный OPENROUTER_API_KEY. Проверь ключ: "
                "https://openrouter.ai/settings/keys"
            )
        if r.status_code == 429:
            raise RuntimeError(
                "лимит запросов OpenRouter (rate limit). "
                "Подожди минуту или пополни баланс — у бесплатных моделей лимиты строгие."
            )
        if r.status_code == 404:
            raise RuntimeError(
                f"модель '{OPENROUTER_MODEL}' не найдена на OpenRouter. "
                "Проверь slug: https://openrouter.ai/models"
            )
        r.raise_for_status()
        data = r.json()

    # OpenRouter может вернуть 200 с полем error (напр. провайдер упал)
    if "error" in data and not data.get("choices"):
        msg = data["error"].get("message", "неизвестная ошибка OpenRouter")
        raise RuntimeError(f"OpenRouter: {msg}")

    content = data["choices"][0]["message"]["content"].strip()

    # На случай если модель всё-таки вернула ```json ... ```
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()

    parsed = json.loads(content)
    return {
        "title": _strip_markdown((parsed.get("title") or "Без назви")).strip()[:200],
        "summary": _strip_markdown((parsed.get("summary") or "")).strip(),
        "tag": _strip_markdown((parsed.get("tag") or "Інше")).strip(),
    }


def _chunks(text: str, size: int = 1900):
    """Notion не любит > 2000 символов в одном rich_text — режем."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _paragraph(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _heading(text: str, level: int = 2) -> dict:
    key = f"heading_{level}"
    return {
        "object": "block",
        "type": key,
        key: {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def create_notion_page(structured: dict, transcription: str) -> str:
    """
    Создаёт страницу в Notion. Для длинных аудио разбивает контент
    на батчи по 90 блоков и дописывает через blocks.children.append.
    """
    title = structured["title"]
    summary = structured["summary"]
    tag = structured["tag"]
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Свойства (колонки таблицы). rich_text у 'Суть' может содержать до 100 частей.
    properties = {
        PROP_TITLE: {"title": [{"type": "text", "text": {"content": title}}]},
        PROP_ESSENCE: {
            "rich_text": [
                {"type": "text", "text": {"content": chunk}}
                for chunk in _chunks(summary)
            ][:100]
        },
        PROP_DATE: {"date": {"start": date_str}},
        PROP_TAG: {"select": {"name": tag}},
    }

    # Собираем тело страницы плоским списком блоков (без toggle,
    # чтобы можно было честно батчить длинные транскрипции).
    body: list[dict] = [_heading("Конспект", 2)]
    body.extend(_paragraph(c) for c in _chunks(summary))
    body.append(_heading("Оригинальная транскрипция", 2))
    body.extend(_paragraph(c) for c in _chunks(transcription))

    # Notion: не более 100 блоков за один запрос → берём 90 с запасом.
    batches = list(_batched(body, 90))
    first, rest = batches[0], batches[1:]

    page = notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties=properties,
        children=first,
    )
    page_id = page["id"]

    for batch in rest:
        notion.blocks.children.append(block_id=page_id, children=batch)

    return page.get("url", "")


# =====================================================================
# handlers
# =====================================================================
async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        return await _reject(update)
    await update.message.reply_text(
        "Привет! 👋\n\n"
        "Пришли мне голосовое сообщение (надиктованное или пересланное из другого "
        "чата — например, кусок подкаста), и я:\n"
        "1. Расшифрую его через Whisper.\n"
        "2. Спрошу, есть ли моменты, которые ты хочешь выделить.\n"
        "3. Структурирую всё через Claude.\n"
        "4. Сохраню карточку в твою базу Notion.\n\n"
        "Языки: русский, українська."
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _cleanup_voice(context)
    context.user_data.clear()
    if update.message:
        await update.message.reply_text("Отменил. Пришли новое голосовое, когда будешь готов.")
    return ConversationHandler.END


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Точка входа: пришло голосовое или аудио. Скачиваем и спрашиваем язык."""
    if not _is_allowed(update):
        await _reject(update)
        return ConversationHandler.END

    msg = update.message
    status = await msg.reply_text("🎧 Скачиваю аудио…")

    # Выбираем источник (voice — обычное «кружочек», audio — файл/пересланное)
    if msg.voice:
        tg_file = await msg.voice.get_file()
        ext = ".ogg"
    elif msg.audio:
        tg_file = await msg.audio.get_file()
        ext = Path(msg.audio.file_name or "audio.mp3").suffix or ".mp3"
    elif msg.video_note:
        tg_file = await msg.video_note.get_file()
        ext = ".mp4"
    else:
        await status.edit_text("❌ Не вижу голосового/аудио.")
        return ConversationHandler.END

    voice_path = TMP_DIR / f"{msg.chat_id}_{msg.message_id}{ext}"
    await tg_file.download_to_drive(str(voice_path))

    # Запоминаем путь и координаты статус-сообщения
    context.user_data["voice_path"] = str(voice_path)
    context.user_data["status_chat_id"] = status.chat_id
    context.user_data["status_msg_id"] = status.message_id

    # Если язык задан в конфиге — не спрашиваем, сразу транскрибируем
    if DEFAULT_LANG in ("ru", "uk", "auto"):
        lang = None if DEFAULT_LANG == "auto" else DEFAULT_LANG
        return await _run_transcription(context, status, lang)

    # Иначе спрашиваем язык
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
                InlineKeyboardButton("🇺🇦 Українська", callback_data="lang_uk"),
            ],
            [InlineKeyboardButton("🤖 Определить автоматически", callback_data="lang_auto")],
            [InlineKeyboardButton("❌ Отмена", callback_data="cancel")],
        ]
    )
    await status.edit_text(
        "🎧 Аудио получено. На каком языке говорят?\n"
        "<i>(подсказка языка сильно повышает точность и убирает "
        "случайный перевод на английский)</i>",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
    return CHOOSING_LANGUAGE


async def handle_language_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Пользователь выбрал язык — запускаем транскрипцию."""
    query = update.callback_query
    await query.answer()

    if query.data == "cancel":
        _cleanup_voice(context)
        context.user_data.clear()
        await query.edit_message_text("❌ Отменил.")
        return ConversationHandler.END

    lang_map = {"lang_ru": "ru", "lang_uk": "uk", "lang_auto": None}
    lang = lang_map.get(query.data, None)
    return await _run_transcription(context, query.message, lang)


async def _run_transcription(context: ContextTypes.DEFAULT_TYPE, status_msg,
                             language: str | None) -> int:
    """Общий раннер: транскрибирует уже скачанное аудио и показывает превью."""
    voice_path = Path(context.user_data.get("voice_path", ""))
    if not voice_path.is_file():
        await status_msg.edit_text("❌ Потерял файл аудио. Пришли голосовое заново.")
        context.user_data.clear()
        return ConversationHandler.END

    try:
        transcription = await transcribe(voice_path, status_msg, language)
    except Exception as e:  # noqa: BLE001
        log.exception("Whisper failed")
        await status_msg.edit_text(f"❌ Ошибка транскрипции: {e}")
        _cleanup_voice(context)
        context.user_data.clear()
        return ConversationHandler.END
    finally:
        _cleanup_voice(context)

    if not transcription:
        await status_msg.edit_text("❌ Whisper вернул пустой текст. Попробуй ещё раз.")
        context.user_data.clear()
        return ConversationHandler.END

    context.user_data["transcription"] = transcription

    preview = transcription if len(transcription) <= 900 else transcription[:900] + "…"
    total_chars = len(transcription)
    length_note = ""
    if total_chars > 900:
        length_note = f"\n<i>(показан фрагмент; всего {total_chars} символов, ~{total_chars // 5} слов)</i>\n"
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✍️ Добавить акценты", callback_data="add_highlights")],
            [InlineKeyboardButton("⚡ Обработать как есть", callback_data="process_now")],
            [InlineKeyboardButton("❌ Отмена", callback_data="cancel")],
        ]
    )
    await status_msg.edit_text(
        f"✅ <b>Транскрипция готова</b>\n\n<i>{preview}</i>\n{length_note}\n"
        "Есть моменты, которые хочешь выделить в конспекте?",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
    return WAITING_FOR_HIGHLIGHTS


def _cleanup_voice(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Удаляет скачанный аудиофайл, если ещё лежит."""
    p = context.user_data.get("voice_path")
    if p:
        Path(p).unlink(missing_ok=True)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "add_highlights":
        await query.edit_message_text(
            "✍️ Ок, напиши одним сообщением, что выделить.\n\n"
            "Например: <i>«Подчеркни всё про воронку продаж и вывод про AI-агентов, "
            "остальное можно короче»</i>",
            parse_mode=ParseMode.HTML,
        )
        return WAITING_FOR_HIGHLIGHTS

    if action == "process_now":
        await query.edit_message_text("🧠 Структурирую и сохраняю в Notion…")
        await _finalize(context, highlights=None, chat_id=query.message.chat_id,
                        edit_message_id=query.message.message_id)
        return ConversationHandler.END

    if action == "cancel":
        context.user_data.clear()
        await query.edit_message_text("❌ Отменил.")
        return ConversationHandler.END

    return WAITING_FOR_HIGHLIGHTS


async def handle_highlights_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Пользователь прислал текст с акцентами — обрабатываем."""
    if not _is_allowed(update):
        return ConversationHandler.END

    highlights = (update.message.text or "").strip()
    status = await update.message.reply_text("🧠 Учёл акценты, структурирую и сохраняю в Notion…")
    await _finalize(
        context,
        highlights=highlights,
        chat_id=status.chat_id,
        edit_message_id=status.message_id,
    )
    return ConversationHandler.END


async def _finalize(context: ContextTypes.DEFAULT_TYPE, *, highlights: str | None,
                    chat_id: int, edit_message_id: int) -> None:
    """Общий финал: Claude → Notion → отчёт пользователю."""
    bot = context.bot
    transcription = context.user_data.get("transcription", "")

    async def _edit(text: str, **kw) -> None:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=edit_message_id,
                text=text,
                **kw,
            )
        except Exception:  # noqa: BLE001
            await bot.send_message(chat_id=chat_id, text=text, **kw)

    if not transcription:
        await _edit("❌ Потерял транскрипцию. Пришли голосовое заново.")
        context.user_data.clear()
        return

    try:
        structured = await analyze_with_claude(transcription, highlights)
    except Exception as e:  # noqa: BLE001
        log.exception("Claude/OpenRouter failed")
        await _edit(f"❌ Ошибка анализа (OpenRouter/Claude): {e}")
        context.user_data.clear()
        return

    try:
        page_url = await asyncio.to_thread(create_notion_page, structured, transcription)
    except Exception as e:  # noqa: BLE001
        log.exception("Notion failed")
        await _edit(
            f"❌ Ошибка Notion: {e}\n\n"
            "Проверь: имена колонок в базе, тип поля «Тег» (Select), доступ интеграции к базе."
        )
        context.user_data.clear()
        return

    summary_preview = structured["summary"]
    if len(summary_preview) > 1500:
        summary_preview = summary_preview[:1500] + "…"

    await _edit(
        f"✅ <b>Готово!</b>\n\n"
        f"<b>{structured['title']}</b>\n"
        f"🏷 <code>{structured['tag']}</code>\n\n"
        f"{summary_preview}\n\n"
        f"🔗 <a href=\"{page_url}\">Открыть в Notion</a>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    context.user_data.clear()


# =====================================================================
# main
# =====================================================================
def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[
            MessageHandler(
                filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE,
                handle_voice,
            )
        ],
        states={
            CHOOSING_LANGUAGE: [
                CallbackQueryHandler(handle_language_choice),
            ],
            WAITING_FOR_HIGHLIGHTS: [
                CallbackQueryHandler(handle_callback),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_highlights_text),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_message=False,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(conv)

    log.info("Bot started (chat=%s, stt=%s)", OPENROUTER_MODEL, WHISPER_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
