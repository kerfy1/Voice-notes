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
import html
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

import admin
import db

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

# ---------- доступ и лимиты ----------
# Админы: полный функционал + Notion + админ-панель. ID через запятую (@userinfobot).
_admins = os.getenv("ADMIN_IDS", os.getenv("ALLOWED_USER_IDS", "")).strip()
ADMIN_IDS = {int(x) for x in _admins.split(",") if x.strip()}

# Лимиты обычных пользователей (скользящее окно 7 дней).
# Срабатывает то, что кончится раньше: количество ИЛИ объём.
FREE_LIMITS = {
    "voice_count": int(os.getenv("FREE_VOICE_COUNT", "5")),
    "voice_seconds": int(os.getenv("FREE_VOICE_MINUTES", "31")) * 60,
    "text_count": int(os.getenv("FREE_TEXT_COUNT", "5")),
    "text_chars": int(os.getenv("FREE_TEXT_CHARS", "25000")),
}
# Максимальный размер одного текстового сообщения для обычного пользователя.
FREE_MAX_TEXT_LENGTH = int(os.getenv("FREE_MAX_TEXT_LENGTH", "5000"))
# Максимальная длина одного голосового для обычного пользователя (сек).
FREE_MAX_VOICE_SECONDS = int(os.getenv("FREE_MAX_VOICE_MINUTES", "15")) * 60
# Потолок выжимки для обычного пользователя — чтобы влезала в одно сообщение TG.
FREE_SUMMARY_CHARS = int(os.getenv("FREE_SUMMARY_CHARS", "1200"))

# Язык транскрипции по умолчанию для кнопки-подсказки.
# Пусто → бот всегда спрашивает. Можно задать "ru"/"uk"/"auto", чтобы пропускать вопрос.
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "").strip().lower()

# Минимальная длина текстового сообщения, чтобы делать из него конспект.
# Короткие реплики ("ок", "привет") игнорируем.
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "100"))

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
notion = NotionClient(auth=NOTION_TOKEN) if NOTION_TOKEN else None


# =====================================================================
# helpers
# =====================================================================
def _esc(text: str) -> str:
    """
    Экранирует HTML для Telegram parse_mode=HTML.
    Обязательно для любого пользовательского текста: символы < > & сломают разметку.
    """
    return html.escape(text or "", quote=False)


def _is_admin(update: Update) -> bool:
    u = update.effective_user
    return bool(u and u.id in ADMIN_IDS)


async def _register(update: Update) -> dict:
    """Регистрирует/обновляет пользователя в БД и возвращает его запись."""
    u = update.effective_user
    await asyncio.to_thread(db.upsert_user, u.id, u.username, u.first_name)
    return await asyncio.to_thread(db.get_user, u.id)


def _fmt_left(usage: dict) -> str:
    """Строка «сколько осталось» для обычного пользователя."""
    v_left = max(0, FREE_LIMITS["voice_count"] - usage["voice_count"])
    v_min_left = max(0.0, (FREE_LIMITS["voice_seconds"] - usage["voice_seconds"]) / 60)
    t_left = max(0, FREE_LIMITS["text_count"] - usage["text_count"])
    t_ch_left = max(0, FREE_LIMITS["text_chars"] - usage["text_chars"])
    return (
        f"🗣 Голосовых: {v_left} шт / {v_min_left:.0f} мин\n"
        f"📝 Текстов: {t_left} шт / {t_ch_left} симв."
    )


async def _check_limits(user_id: int, kind: str, amount: float) -> tuple[bool, str]:
    """
    Проверяет лимиты обычного пользователя.
    kind: "voice" (amount = секунды) | "text" (amount = символы).
    Возвращает (можно?, текст_отказа).
    """
    usage = await asyncio.to_thread(db.weekly_usage, user_id)

    if kind == "voice":
        if usage["voice_count"] >= FREE_LIMITS["voice_count"]:
            return False, (
                f"⛔ Недельный лимит голосовых исчерпан "
                f"({FREE_LIMITS['voice_count']} шт).\n\n"
                f"Осталось:\n{_fmt_left(usage)}\n\n"
                "Лимит обновляется автоматически (скользящие 7 дней)."
            )
        if usage["voice_seconds"] + amount > FREE_LIMITS["voice_seconds"]:
            left = (FREE_LIMITS["voice_seconds"] - usage["voice_seconds"]) / 60
            return False, (
                f"⛔ Не хватает минут: это голосовое на {amount/60:.1f} мин, "
                f"а осталось {left:.1f} мин из "
                f"{FREE_LIMITS['voice_seconds']/60:.0f} в неделю.\n\n"
                f"Осталось:\n{_fmt_left(usage)}"
            )
    else:  # text
        if usage["text_count"] >= FREE_LIMITS["text_count"]:
            return False, (
                f"⛔ Недельный лимит текстов исчерпан "
                f"({FREE_LIMITS['text_count']} шт).\n\n"
                f"Осталось:\n{_fmt_left(usage)}\n\n"
                "Лимит обновляется автоматически (скользящие 7 дней)."
            )
        if usage["text_chars"] + amount > FREE_LIMITS["text_chars"]:
            left = FREE_LIMITS["text_chars"] - usage["text_chars"]
            return False, (
                f"⛔ Не хватает символов: в тексте {int(amount)}, "
                f"а осталось {left} из {FREE_LIMITS['text_chars']} в неделю.\n\n"
                f"Осталось:\n{_fmt_left(usage)}"
            )

    return True, ""


async def _gate(update: Update, kind: str, amount: float) -> tuple[bool, bool, str]:
    """
    Единая проверка на входе.
    Возвращает (пропустить?, is_admin, текст_отказа).
    """
    user = await _register(update)
    is_admin = _is_admin(update)

    if is_admin or user["status"] == "unlimited":
        return True, is_admin, ""

    if user["status"] == "blocked":
        return False, False, "⛔ Доступ к боту ограничен."

    ok, reason = await _check_limits(user["user_id"], kind, amount)
    return ok, False, reason


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


async def analyze_with_claude(source_text: str, highlights: str | None,
                              source: str = "voice", concise: bool = False) -> dict:
    """
    Отправляет исходный текст (+ акценты) в Claude через OpenRouter, ждёт JSON.
    source:  "voice" — расшифровка голосового, "text" — текст напрямую.
    concise: True для обычных пользователей — краткая выжимка в одно сообщение TG.
    """
    origin = (
        "расшифровок голосовых сообщений"
        if source == "voice"
        else "текстов, которые пользователь присылает напрямую "
             "(свои заметки, пересланные посты, статьи, выдержки)"
    )
    if concise:
        summary_rule = (
            f"  - summary: КРАТКАЯ выжимка, строго не длиннее "
            f"{FREE_SUMMARY_CHARS} символов. Только самое важное: "
            "5-8 ключевых тезисов маркированным списком (`- `). "
            "Без вступлений и воды. Тот же язык, что и оригинал.\n"
        )
    else:
        summary_rule = (
            "  - summary: структурированный конспект. Выдели ключевые тезисы "
            "маркированным списком (используй `- ` для пунктов), при необходимости — "
            "подсекции. Пиши по делу, без воды. Тот же язык, что и оригинал.\n"
        )

    system_prompt = (
        f"Ты помощник, который делает структурированные конспекты из "
        f"{origin} на русском или украинском языке.\n\n"
        "Проанализируй текст и верни строго валидный JSON без markdown-обрамления "
        "и без лишних комментариев, с полями:\n"
        "  - title: короткий заголовок конспекта (до 60 символов), "
        "на том же языке, что и оригинал.\n"
        + summary_rule +
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

    label = "Транскрипция" if source == "voice" else "Исходный текст"
    user_content = f"{label}:\n\"\"\"\n{source_text}\n\"\"\""
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


def create_notion_page(structured: dict, source_text: str, source: str = "voice") -> str:
    """
    Создаёт страницу в Notion. Для длинных аудио разбивает контент
    на батчи по 90 блоков и дописывает через blocks.children.append.
    source: "voice" | "text" — влияет только на заголовок блока с оригиналом.
    """
    if notion is None or not NOTION_DATABASE_ID:
        raise RuntimeError(
            "Notion не настроен: заполни NOTION_TOKEN и NOTION_DATABASE_ID "
            "в файле api_key."
        )
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
    origin_heading = (
        "Оригинальная транскрипция" if source == "voice" else "Исходный текст"
    )
    body: list[dict] = [_heading("Конспект", 2)]
    body.extend(_paragraph(c) for c in _chunks(summary))
    body.append(_heading(origin_heading, 2))
    body.extend(_paragraph(c) for c in _chunks(source_text))

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
    await _register(update)
    await update.message.reply_text(
        "Привет! 👋\n\n"
        "Я делаю структурированные конспекты.\n\n"
        "🗣 <b>Голосовое</b> (надиктованное или пересланное — например, кусок "
        "подкаста): расшифрую, потом структурирую.\n\n"
        "📝 <b>Текст</b> (своя заметка, пересланный пост, выдержка из статьи): "
        "структурирую, сделаю выжимку.\n\n"
        "Дополнительно можешь указать моменты, которые хочешь выделить для заметки.\n\n"
        "Языки: українська, русский.",
        parse_mode=ParseMode.HTML,
    )


async def cmd_limits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает пользователю остаток недельной квоты."""
    user = await _register(update)

    if _is_admin(update):
        await update.message.reply_text("⭐ У тебя админский доступ — без лимитов.")
        return
    if user["status"] == "unlimited":
        await update.message.reply_text("⭐ У тебя безлимитный доступ.")
        return
    if user["status"] == "blocked":
        await update.message.reply_text("⛔ Доступ к боту ограничен.")
        return

    usage = await asyncio.to_thread(db.weekly_usage, user["user_id"])
    await update.message.reply_text(
        "📊 <b>Осталось на неделю</b>\n\n"
        f"{_esc(_fmt_left(usage))}\n\n"
        f"<i>Лимит: {FREE_LIMITS['voice_count']} голосовых "
        f"({FREE_LIMITS['voice_seconds']//60} мин) и "
        f"{FREE_LIMITS['text_count']} текстов "
        f"({FREE_LIMITS['text_chars']} симв.) за 7 дней.\n"
        "Окно скользящее — квота восстанавливается постепенно.</i>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    _cleanup_voice(context)
    context.user_data.clear()
    if update.message:
        await update.message.reply_text("Отменил. Пришли новое голосовое, когда будешь готов.")
    return ConversationHandler.END


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Точка входа: голосовое/аудио.
    Скачиваем → узнаём длительность → проверяем лимиты → спрашиваем язык.
    Длительность проверяем ДО транскрипции, чтобы не тратить деньги впустую.
    """
    msg = update.message
    is_admin = _is_admin(update)

    # Быстрая длительность из метаданных Telegram (без скачивания)
    tg_duration = 0
    if msg.voice:
        tg_duration = msg.voice.duration or 0
    elif msg.audio:
        tg_duration = msg.audio.duration or 0
    elif msg.video_note:
        tg_duration = msg.video_note.duration or 0

    # Потолок на одно голосовое для обычных пользователей
    if not is_admin and tg_duration > FREE_MAX_VOICE_SECONDS:
        await msg.reply_text(
            f"⛔ Слишком длинное аудио: {tg_duration/60:.1f} мин.\n"
            f"Максимум за раз — {FREE_MAX_VOICE_SECONDS/60:.0f} мин."
        )
        return ConversationHandler.END

    # Проверка лимитов (регистрирует пользователя в БД)
    ok, is_admin, reason = await _gate(update, "voice", tg_duration)
    if not ok:
        await msg.reply_text(reason)
        return ConversationHandler.END

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
    try:
        await tg_file.download_to_drive(str(voice_path))
    except Exception as e:  # noqa: BLE001
        log.exception("Download failed")
        await status.edit_text(
            "❌ Не смог скачать файл. Telegram отдаёт боту файлы до 20 МБ — "
            "возможно, аудио слишком большое."
        )
        return ConversationHandler.END

    user = update.effective_user
    db_user = await asyncio.to_thread(db.get_user, user.id)

    context.user_data.update({
        "voice_path": str(voice_path),
        "status_chat_id": status.chat_id,
        "status_msg_id": status.message_id,
        "is_admin": is_admin,
        "user_id": user.id,
        "unlimited": bool(db_user and db_user["status"] == "unlimited"),
        "charge": tg_duration,   # спишем секунды после успешной обработки
    })

    # Если язык задан в конфиге — не спрашиваем, сразу транскрибируем
    if DEFAULT_LANG in ("ru", "uk", "auto"):
        lang = None if DEFAULT_LANG == "auto" else DEFAULT_LANG
        return await _run_transcription(context, status, lang)

    # Иначе спрашиваем язык
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🇺🇦 Українська", callback_data="lang_uk"),
                InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
            ],
            [InlineKeyboardButton("🤖 Определить автоматически", callback_data="lang_auto")],
            [InlineKeyboardButton("❌ Отмена", callback_data="cancel")],
        ]
    )
    await status.edit_text(
        "🎧 Аудио получено. На каком языке говорят?\n"
        "<i>(подсказка языка повышает точность расшифровки)</i>",
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
        if context.user_data.get("is_admin"):
            await status_msg.edit_text(f"❌ Ошибка транскрипции: {e}")
        else:
            await status_msg.edit_text(
                "❌ Не получилось расшифровать аудио. Попробуй ещё раз."
            )
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
    context.user_data["source"] = "voice"

    is_admin = context.user_data.get("is_admin", False)

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✍️ Добавить акценты", callback_data="add_highlights")],
            [InlineKeyboardButton("⚡ Обработать как есть", callback_data="process_now")],
            [InlineKeyboardButton("❌ Отмена", callback_data="cancel")],
        ]
    )

    if is_admin:
        # Админу показываем транскрипцию
        preview = transcription if len(transcription) <= 900 else transcription[:900] + "…"
        total_chars = len(transcription)
        length_note = ""
        if total_chars > 900:
            length_note = (
                f"\n<i>(показан фрагмент; всего {total_chars} символов, "
                f"~{total_chars // 5} слов)</i>\n"
            )
        body = (
            f"✅ <b>Транскрипция готова</b>\n\n"
            f"<i>{_esc(preview)}</i>\n{length_note}\n"
        )
    else:
        # Обычным — без транскрипции
        body = "✅ <b>Расшифровал.</b>\n\n"

    await status_msg.edit_text(
        body + "Есть моменты, которые хочешь выделить в конспекте?",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
    return WAITING_FOR_HIGHLIGHTS


def _cleanup_voice(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Удаляет скачанный аудиофайл, если ещё лежит."""
    p = context.user_data.get("voice_path")
    if p:
        Path(p).unlink(missing_ok=True)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Точка входа: пришёл обычный текст (своя заметка, пересланный пост, статья).
    Пропускаем Whisper и выбор языка — сразу спрашиваем про акценты.
    """
    text = (update.message.text or "").strip()
    is_admin = _is_admin(update)

    if len(text) < MIN_TEXT_LENGTH:
        await update.message.reply_text(
            f"🤔 Слишком короткий текст для конспекта "
            f"(нужно хотя бы {MIN_TEXT_LENGTH} символов).\n"
            "Пришли голосовое или текст подлиннее."
        )
        return ConversationHandler.END

    # Потолок на одно сообщение для обычных пользователей
    if not is_admin and len(text) > FREE_MAX_TEXT_LENGTH:
        await update.message.reply_text(
            f"⛔ Слишком длинный текст: {len(text)} символов.\n"
            f"Максимум за раз — {FREE_MAX_TEXT_LENGTH}."
        )
        return ConversationHandler.END

    # Проверка лимитов (регистрирует пользователя в БД)
    ok, is_admin, reason = await _gate(update, "text", len(text))
    if not ok:
        await update.message.reply_text(reason)
        return ConversationHandler.END

    user = update.effective_user
    db_user = await asyncio.to_thread(db.get_user, user.id)

    context.user_data.update({
        "transcription": text,
        "source": "text",
        "is_admin": is_admin,
        "user_id": user.id,
        "unlimited": bool(db_user and db_user["status"] == "unlimited"),
        "charge": len(text),   # спишем символы после успешной обработки
    })

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✍️ Добавить акценты", callback_data="add_highlights")],
            [InlineKeyboardButton("⚡ Обработать как есть", callback_data="process_now")],
            [InlineKeyboardButton("❌ Отмена", callback_data="cancel")],
        ]
    )

    if is_admin:
        preview = text if len(text) <= 900 else text[:900] + "…"
        length_note = ""
        if len(text) > 900:
            length_note = (
                f"\n<i>(показан фрагмент; всего {len(text)} символов)</i>\n"
            )
        body = f"📝 <b>Текст получен</b>\n\n<i>{_esc(preview)}</i>\n{length_note}\n"
    else:
        body = f"📝 <b>Текст получен</b> ({len(text)} символов)\n\n"

    status = await update.message.reply_text(
        body + "Есть моменты, которые хочешь выделить в конспекте?",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )
    context.user_data["status_chat_id"] = status.chat_id
    context.user_data["status_msg_id"] = status.message_id
    return WAITING_FOR_HIGHLIGHTS


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
    highlights = (update.message.text or "").strip()[:1000]
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
    """
    Общий финал: Claude → (Notion для админа | только чат для обычных) → отчёт.
    Здесь же списывается квота обычного пользователя.
    """
    bot = context.bot
    source_text = context.user_data.get("transcription", "")
    source = context.user_data.get("source", "voice")   # "voice" | "text"
    is_admin = context.user_data.get("is_admin", False)
    user_id = context.user_data.get("user_id")
    # Сколько списать: секунды для голосового, символы для текста
    charge = context.user_data.get("charge", 0)
    unlimited = context.user_data.get("unlimited", False)

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

    if not source_text:
        await _edit("❌ Потерял исходный текст. Пришли голосовое или текст заново.")
        context.user_data.clear()
        return

    try:
        structured = await analyze_with_claude(
            source_text, highlights, source, concise=not is_admin
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Claude/OpenRouter failed")
        await _edit(f"❌ Ошибка анализа: {e}" if is_admin
                    else "❌ Не получилось обработать. Попробуй ещё раз чуть позже.")
        context.user_data.clear()
        return

    # ---------- обычный пользователь: выжимка только в чат, без Notion ----------
    if not is_admin:
        # списываем квоту (безлимитным — не списываем)
        if user_id and not unlimited:
            await asyncio.to_thread(db.add_usage, user_id, source, charge)

        summary = structured["summary"]
        if len(summary) > FREE_SUMMARY_CHARS:
            summary = summary[:FREE_SUMMARY_CHARS].rsplit("\n", 1)[0] + "…"

        text = (
            f"✅ <b>{_esc(structured['title'])}</b>\n"
            f"🏷 {_esc(structured['tag'])}\n\n"
            f"{_esc(summary)}"
        )

        # Хвост с остатком лимита
        if user_id and not unlimited:
            usage = await asyncio.to_thread(db.weekly_usage, user_id)
            text += f"\n\n<i>Осталось на неделю:\n{_esc(_fmt_left(usage))}</i>"

        # Страховка: TG режет на 4096
        if len(text) > 4000:
            text = text[:4000] + "…"

        await _edit(text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        context.user_data.clear()
        return

    # ---------- админ: полный конспект + Notion ----------
    try:
        page_url = await asyncio.to_thread(
            create_notion_page, structured, source_text, source
        )
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
        f"<b>{_esc(structured['title'])}</b>\n"
        f"🏷 <code>{_esc(structured['tag'])}</code>\n\n"
        f"{_esc(summary_preview)}\n\n"
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
            ),
            # Текст как второй вход: своя заметка, пересланный пост, статья.
            # Срабатывает только вне диалога — внутри WAITING_FOR_HIGHLIGHTS
            # текст перехватывается как «акценты».
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                handle_text,
            ),
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
    app.add_handler(CommandHandler("limits", cmd_limits))
    app.add_handler(CommandHandler("cancel", cmd_cancel))

    # ВАЖНО: админка регистрируется ДО основного диалога.
    # Иначе ввод админа (поиск пользователя) уйдёт в конспект как текст.
    app.add_handler(admin.build_admin_handler(ADMIN_IDS, FREE_LIMITS))
    app.add_handler(conv)

    log.info(
        "Bot started (chat=%s, stt=%s, admins=%s)",
        OPENROUTER_MODEL, WHISPER_MODEL, sorted(ADMIN_IDS) or "нет!",
    )
    if not ADMIN_IDS:
        log.warning("ADMIN_IDS пуст — админ-панель недоступна, Notion не работает ни для кого!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    db.init_db()
    main()
