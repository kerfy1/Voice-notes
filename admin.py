"""
Админ-панель: /admin

Функции:
  📊 Статистика     — пользователи, расход за неделю и за всё время
  🏆 Топ активных   — кто больше всех пользуется
  👥 Пользователи   — последние активные
  🔍 Найти          — по user_id или @username → карточка с действиями
  🔄 Сбросить всем  — обнулить лимиты всех пользователей

Карточка пользователя: блокировка, безлимит, сброс лимитов.

Регистрируется ПЕРЕД основным ConversationHandler, чтобы перехватывать
ввод админа (поиск пользователя) раньше, чем текст уйдёт в конспект.
"""

import asyncio
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import db

log = logging.getLogger("voicebot.admin")

# состояния
ADMIN_MENU = 100
ADMIN_AWAIT_QUERY = 101

# Заполняется из bot.py при регистрации
ADMIN_IDS: set[int] = set()
LIMITS: dict = {}


def _is_admin(update: Update) -> bool:
    u = update.effective_user
    return bool(u and u.id in ADMIN_IDS)


def _menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Статистика", callback_data="adm_stats"),
                InlineKeyboardButton("🏆 Топ активных", callback_data="adm_top"),
            ],
            [
                InlineKeyboardButton("👥 Пользователи", callback_data="adm_users"),
                InlineKeyboardButton("🔍 Найти", callback_data="adm_find"),
            ],
            [InlineKeyboardButton("🔄 Сбросить лимиты всем", callback_data="adm_reset_all")],
            [InlineKeyboardButton("✖️ Закрыть", callback_data="adm_close")],
        ]
    )


def _back_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⬅️ Назад", callback_data="adm_menu")]]
    )


def _user_label(u: dict) -> str:
    name = u.get("first_name") or "—"
    uname = f"@{u['username']}" if u.get("username") else f"id{u['user_id']}"
    return f"{name} ({uname})"


def _status_emoji(status: str) -> str:
    return {"active": "✅", "blocked": "🚫", "unlimited": "⭐"}.get(status, "❓")


def _user_card_kb(user_id: int, status: str) -> InlineKeyboardMarkup:
    rows = []
    if status == "blocked":
        rows.append([InlineKeyboardButton("✅ Разблокировать",
                                          callback_data=f"adm_active_{user_id}")])
    else:
        rows.append([InlineKeyboardButton("🚫 Заблокировать",
                                          callback_data=f"adm_block_{user_id}")])
    if status == "unlimited":
        rows.append([InlineKeyboardButton("↩️ Снять безлимит",
                                          callback_data=f"adm_active_{user_id}")])
    else:
        rows.append([InlineKeyboardButton("⭐ Выдать безлимит",
                                          callback_data=f"adm_unlim_{user_id}")])
    rows.append([InlineKeyboardButton("🔄 Сбросить лимиты",
                                      callback_data=f"adm_reset_{user_id}")])
    rows.append([InlineKeyboardButton("⬅️ В меню", callback_data="adm_menu")])
    return InlineKeyboardMarkup(rows)


async def _render_user_card(user: dict) -> tuple[str, InlineKeyboardMarkup]:
    usage = await asyncio.to_thread(db.weekly_usage, user["user_id"])
    status = user["status"]

    if status == "unlimited":
        limits_line = "⭐ Безлимит"
    elif status == "blocked":
        limits_line = "🚫 Заблокирован"
    else:
        limits_line = (
            f"🗣 {usage['voice_count']}/{LIMITS['voice_count']} шт, "
            f"{usage['voice_seconds']/60:.1f}/{LIMITS['voice_seconds']/60:.0f} мин\n"
            f"📝 {usage['text_count']}/{LIMITS['text_count']} шт, "
            f"{usage['text_chars']}/{LIMITS['text_chars']} симв."
        )

    text = (
        f"{_status_emoji(status)} <b>{_user_label(user)}</b>\n"
        f"<code>{user['user_id']}</code>\n\n"
        f"Первый контакт: {user['created_at']}\n"
        f"Последний раз: {user['last_seen']}\n\n"
        f"<b>Расход за 7 дней:</b>\n{limits_line}"
    )
    return text, _user_card_kb(user["user_id"], status)


# ------------------------------------------------------------------ handlers
async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not _is_admin(update):
        return ConversationHandler.END
    await update.message.reply_text(
        "🛠 <b>Админ-панель</b>", parse_mode=ParseMode.HTML, reply_markup=_menu_kb()
    )
    return ADMIN_MENU


async def on_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "adm_close":
        await query.edit_message_text("Панель закрыта.")
        return ConversationHandler.END

    if data == "adm_menu":
        await query.edit_message_text(
            "🛠 <b>Админ-панель</b>", parse_mode=ParseMode.HTML, reply_markup=_menu_kb()
        )
        return ADMIN_MENU

    # ---------- статистика
    if data == "adm_stats":
        s = await asyncio.to_thread(db.get_stats)
        w, a = s["week"], s["all_time"]
        text = (
            "📊 <b>Статистика</b>\n\n"
            f"<b>Пользователи</b>\n"
            f"Всего: {s['total_users']}  (+{s['new_week']} за неделю)\n"
            f"Активны за 7 дней: {s['active_week']}\n"
            f"⭐ Безлимит: {s['unlimited']}   🚫 Заблокировано: {s['blocked']}\n\n"
            f"<b>За 7 дней</b>\n"
            f"🗣 Голосовых: {w['voice_count']} ({w['voice_minutes']:.0f} мин)\n"
            f"📝 Текстов: {w['text_count']} ({w['text_chars']:,} симв.)\n\n"
            f"<b>За всё время</b>\n"
            f"🗣 Голосовых: {a['voice_count']} ({a['voice_minutes']:.0f} мин)\n"
            f"📝 Текстов: {a['text_count']}"
        ).replace(",", " ")
        await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                      reply_markup=_back_kb())
        return ADMIN_MENU

    # ---------- топ активных
    if data == "adm_top":
        rows = await asyncio.to_thread(db.top_users, 10)
        if not rows:
            text = "🏆 <b>Топ за 7 дней</b>\n\nПока никто не пользовался."
        else:
            lines = ["🏆 <b>Топ за 7 дней</b>\n"]
            for i, r in enumerate(rows, 1):
                lines.append(
                    f"{i}. {_status_emoji(r['status'])} {_user_label(r)}\n"
                    f"    🗣 {r['voice_count']} ({r['voice_seconds']/60:.0f} мин)  "
                    f"📝 {r['text_count']}"
                )
            text = "\n".join(lines)
        await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                      reply_markup=_back_kb())
        return ADMIN_MENU

    # ---------- список пользователей
    if data == "adm_users":
        users = await asyncio.to_thread(db.list_users, 15, 0)
        total = await asyncio.to_thread(db.count_users)
        if not users:
            text = "👥 Пользователей пока нет."
        else:
            lines = [f"👥 <b>Последние активные</b> (всего {total})\n"]
            for u in users:
                lines.append(
                    f"{_status_emoji(u['status'])} {_user_label(u)} — "
                    f"<code>{u['user_id']}</code>"
                )
            lines.append("\n<i>Чтобы управлять — нажми «Найти» и введи id/@username.</i>")
            text = "\n".join(lines)
        await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                      reply_markup=_back_kb())
        return ADMIN_MENU

    # ---------- поиск
    if data == "adm_find":
        await query.edit_message_text(
            "🔍 Введи <b>user_id</b> или <b>@username</b> пользователя:",
            parse_mode=ParseMode.HTML,
            reply_markup=_back_kb(),
        )
        return ADMIN_AWAIT_QUERY

    # ---------- сброс всем
    if data == "adm_reset_all":
        await query.edit_message_text(
            "⚠️ Обнулить лимиты <b>всем</b> пользователям?",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Да, сбросить", callback_data="adm_reset_all_yes")],
                [InlineKeyboardButton("⬅️ Отмена", callback_data="adm_menu")],
            ]),
        )
        return ADMIN_MENU

    if data == "adm_reset_all_yes":
        n = await asyncio.to_thread(db.reset_all_usage)
        await query.edit_message_text(
            f"🔄 Лимиты сброшены. Удалено записей: {n}",
            reply_markup=_back_kb(),
        )
        return ADMIN_MENU

    # ---------- действия над конкретным пользователем
    for prefix, status in (("adm_block_", "blocked"),
                           ("adm_unlim_", "unlimited"),
                           ("adm_active_", "active")):
        if data.startswith(prefix):
            uid = int(data[len(prefix):])
            await asyncio.to_thread(db.set_status, uid, status)
            user = await asyncio.to_thread(db.get_user, uid)
            text, kb = await _render_user_card(user)
            await query.edit_message_text(text, parse_mode=ParseMode.HTML,
                                          reply_markup=kb)
            return ADMIN_MENU

    if data.startswith("adm_reset_"):
        uid = int(data[len("adm_reset_"):])
        await asyncio.to_thread(db.reset_usage, uid)
        user = await asyncio.to_thread(db.get_user, uid)
        text, kb = await _render_user_card(user)
        await query.edit_message_text("🔄 Лимиты сброшены.\n\n" + text,
                                      parse_mode=ParseMode.HTML, reply_markup=kb)
        return ADMIN_MENU

    return ADMIN_MENU


async def on_search_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Админ ввёл id или @username."""
    if not _is_admin(update):
        return ConversationHandler.END

    q = (update.message.text or "").strip()
    user = await asyncio.to_thread(db.find_user, q)
    if not user:
        await update.message.reply_text(
            f"❌ Не нашёл пользователя «{q}».\n"
            "Учти: пользователь появляется в базе только после первого сообщения боту.",
            reply_markup=_back_kb(),
        )
        return ADMIN_MENU

    text, kb = await _render_user_card(user)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)
    return ADMIN_MENU


async def cmd_admin_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Панель закрыта.")
    return ConversationHandler.END


def build_admin_handler(admin_ids: set[int], limits: dict) -> ConversationHandler:
    """Собирает ConversationHandler админки. Вызывать из bot.py."""
    ADMIN_IDS.clear()
    ADMIN_IDS.update(admin_ids)
    LIMITS.clear()
    LIMITS.update(limits)

    return ConversationHandler(
        entry_points=[CommandHandler("admin", cmd_admin)],
        states={
            ADMIN_MENU: [CallbackQueryHandler(on_menu_click, pattern=r"^adm_")],
            ADMIN_AWAIT_QUERY: [
                CallbackQueryHandler(on_menu_click, pattern=r"^adm_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, on_search_input),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_admin_cancel)],
        per_message=False,
    )
