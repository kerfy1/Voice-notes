"""
Слой базы данных: пользователи, учёт использования, лимиты.

SQLite, синхронные функции — вызываются из бота через asyncio.to_thread.
Файл БД лежит рядом со скриптом (bot.db), в systemd-сетапе это /opt/voicebot/app.

Таблицы:
  users — кто пользовался ботом, их статус (active / blocked / unlimited)
  usage — журнал операций: тип (voice/text), объём (секунды/символы), время
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "bot.db"

# SQLite не любит многопоточную запись — сериализуем через lock.
_lock = threading.Lock()


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _lock, _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id     INTEGER PRIMARY KEY,
                username    TEXT,
                first_name  TEXT,
                status      TEXT NOT NULL DEFAULT 'active',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS usage (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                kind        TEXT NOT NULL,           -- 'voice' | 'text'
                amount      REAL NOT NULL DEFAULT 0, -- секунды (voice) | символы (text)
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_usage_user_time
                ON usage(user_id, created_at);
            """
        )


# ---------------------------------------------------------------- users
def upsert_user(user_id: int, username: str | None, first_name: str | None) -> None:
    """Регистрирует пользователя при первом контакте, обновляет last_seen."""
    with _lock, _conn() as con:
        con.execute(
            """
            INSERT INTO users (user_id, username, first_name)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username   = excluded.username,
                first_name = excluded.first_name,
                last_seen  = datetime('now')
            """,
            (user_id, username, first_name),
        )


def get_user(user_id: int) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def find_user(query: str) -> dict | None:
    """Ищет по user_id или по username (с @ или без)."""
    q = query.strip().lstrip("@")
    with _conn() as con:
        if q.isdigit():
            row = con.execute(
                "SELECT * FROM users WHERE user_id = ?", (int(q),)
            ).fetchone()
        else:
            row = con.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (q,)
            ).fetchone()
    return dict(row) if row else None


def set_status(user_id: int, status: str) -> None:
    """status: 'active' | 'blocked' | 'unlimited'"""
    with _lock, _conn() as con:
        con.execute(
            "UPDATE users SET status = ? WHERE user_id = ?", (status, user_id)
        )


def list_users(limit: int = 10, offset: int = 0) -> list[dict]:
    """Пользователи, недавно активные — первыми."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT * FROM users
            ORDER BY last_seen DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def count_users() -> int:
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# ---------------------------------------------------------------- usage
def add_usage(user_id: int, kind: str, amount: float) -> None:
    """kind: 'voice' (amount = секунды) | 'text' (amount = символы)"""
    with _lock, _conn() as con:
        con.execute(
            "INSERT INTO usage (user_id, kind, amount) VALUES (?, ?, ?)",
            (user_id, kind, amount),
        )


def weekly_usage(user_id: int) -> dict:
    """Расход за скользящие 7 дней."""
    with _conn() as con:
        row = con.execute(
            """
            SELECT
                COALESCE(SUM(kind = 'voice'), 0)                        AS voice_count,
                COALESCE(SUM(CASE WHEN kind='voice' THEN amount END),0) AS voice_seconds,
                COALESCE(SUM(kind = 'text'), 0)                         AS text_count,
                COALESCE(SUM(CASE WHEN kind='text'  THEN amount END),0) AS text_chars
            FROM usage
            WHERE user_id = ?
              AND created_at >= datetime('now', '-7 days')
            """,
            (user_id,),
        ).fetchone()
    return {
        "voice_count": int(row["voice_count"]),
        "voice_seconds": float(row["voice_seconds"]),
        "text_count": int(row["text_count"]),
        "text_chars": int(row["text_chars"]),
    }


def reset_usage(user_id: int) -> int:
    """Стирает историю расхода конкретного пользователя. Возвращает кол-во записей."""
    with _lock, _conn() as con:
        cur = con.execute("DELETE FROM usage WHERE user_id = ?", (user_id,))
        return cur.rowcount


def reset_all_usage() -> int:
    with _lock, _conn() as con:
        cur = con.execute("DELETE FROM usage")
        return cur.rowcount


# ---------------------------------------------------------------- stats
def get_stats() -> dict:
    with _conn() as con:
        total_users = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        blocked = con.execute(
            "SELECT COUNT(*) FROM users WHERE status = 'blocked'"
        ).fetchone()[0]
        unlimited = con.execute(
            "SELECT COUNT(*) FROM users WHERE status = 'unlimited'"
        ).fetchone()[0]
        active_week = con.execute(
            """
            SELECT COUNT(DISTINCT user_id) FROM usage
            WHERE created_at >= datetime('now', '-7 days')
            """
        ).fetchone()[0]
        row_week = con.execute(
            """
            SELECT
                COALESCE(SUM(kind='voice'), 0)                          AS voice_count,
                COALESCE(SUM(CASE WHEN kind='voice' THEN amount END),0) AS voice_seconds,
                COALESCE(SUM(kind='text'), 0)                           AS text_count,
                COALESCE(SUM(CASE WHEN kind='text'  THEN amount END),0) AS text_chars
            FROM usage
            WHERE created_at >= datetime('now', '-7 days')
            """
        ).fetchone()
        row_all = con.execute(
            """
            SELECT
                COALESCE(SUM(kind='voice'), 0)                          AS voice_count,
                COALESCE(SUM(CASE WHEN kind='voice' THEN amount END),0) AS voice_seconds,
                COALESCE(SUM(kind='text'), 0)                           AS text_count
            FROM usage
            """
        ).fetchone()
        new_week = con.execute(
            """
            SELECT COUNT(*) FROM users
            WHERE created_at >= datetime('now', '-7 days')
            """
        ).fetchone()[0]

    return {
        "total_users": total_users,
        "new_week": new_week,
        "blocked": blocked,
        "unlimited": unlimited,
        "active_week": active_week,
        "week": {
            "voice_count": int(row_week["voice_count"]),
            "voice_minutes": float(row_week["voice_seconds"]) / 60,
            "text_count": int(row_week["text_count"]),
            "text_chars": int(row_week["text_chars"]),
        },
        "all_time": {
            "voice_count": int(row_all["voice_count"]),
            "voice_minutes": float(row_all["voice_seconds"]) / 60,
            "text_count": int(row_all["text_count"]),
        },
    }


def top_users(limit: int = 10) -> list[dict]:
    """Самые активные за неделю."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT
                u.user_id, u.username, u.first_name, u.status,
                COALESCE(SUM(us.kind='voice'), 0)                          AS voice_count,
                COALESCE(SUM(CASE WHEN us.kind='voice' THEN us.amount END),0) AS voice_seconds,
                COALESCE(SUM(us.kind='text'), 0)                           AS text_count
            FROM users u
            JOIN usage us ON us.user_id = u.user_id
            WHERE us.created_at >= datetime('now', '-7 days')
            GROUP BY u.user_id
            ORDER BY (voice_count + text_count) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
