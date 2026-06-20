from __future__ import annotations

import logging
import sqlite3
from typing import Any

from .config import (
    DB_CHARSET,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    SQLITE_DB_PATH,
)


def _connect_mysql() -> Any:
    try:
        import pymysql
    except ImportError as exc:
        raise ImportError("pymysql is required for MySQL support. Install it or rely on the SQLite fallback.") from exc

    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset=DB_CHARSET,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def _connect_sqlite() -> sqlite3.Connection:
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(SQLITE_DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def get_connection() -> Any:
    try:
        return _connect_mysql()
    except Exception:
        logging.warning("MySQL is unavailable; using local SQLite fallback at %s.", SQLITE_DB_PATH)
        return _connect_sqlite()


def initialize_database() -> None:
    connection = get_connection()
    cursor = connection.cursor()
    is_sqlite = isinstance(connection, sqlite3.Connection)
    primary_key = "INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "INTEGER PRIMARY KEY AUTO_INCREMENT"

    create_table = {
        "users": (
            f"CREATE TABLE IF NOT EXISTS users ("
            f"id {primary_key}, "
            "name TEXT, "
            "email TEXT, "
            "persona TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        ),
        "interactions": (
            f"CREATE TABLE IF NOT EXISTS interactions ("
            f"id {primary_key}, "
            "user_id INTEGER, "
            "query TEXT, "
            "response TEXT, "
            "persona TEXT, "
            "intent TEXT, "
            "confidence REAL, "
            "context TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        ),
        "escalations": (
            f"CREATE TABLE IF NOT EXISTS escalations ("
            f"id {primary_key}, "
            "interaction_id INTEGER, "
            "reason TEXT, "
            "metadata TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ")"
        ),
    }

    for statement in create_table.values():
        cursor.execute(statement)

    if is_sqlite:
        connection.commit()
    cursor.close()
    connection.close()


def log_interaction(
    query: str,
    response: str,
    persona: str,
    intent: str,
    confidence: float,
    context: str | None = None,
    user_id: int | None = None,
) -> int:
    connection = get_connection()
    cursor = connection.cursor()

    is_sqlite = isinstance(connection, sqlite3.Connection)
    placeholder = "?" if is_sqlite else "%s"

    insert = (
        "INSERT INTO interactions (user_id, query, response, persona, intent, confidence, context) "
        f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
    )
    values = (user_id, query, response, persona, intent, confidence, context)
    cursor.execute(insert, values)
    interaction_id = cursor.lastrowid

    if is_sqlite:
        connection.commit()

    cursor.close()
    connection.close()
    return interaction_id


def log_escalation(interaction_id: int, reason: str, metadata: str | None = None) -> int:
    connection = get_connection()
    cursor = connection.cursor()

    is_sqlite = isinstance(connection, sqlite3.Connection)
    placeholder = "?" if is_sqlite else "%s"

    insert = (
        "INSERT INTO escalations (interaction_id, reason, metadata) "
        f"VALUES ({placeholder}, {placeholder}, {placeholder})"
    )
    values = (interaction_id, reason, metadata)
    cursor.execute(insert, values)
    escalation_id = cursor.lastrowid

    if is_sqlite:
        connection.commit()

    cursor.close()
    connection.close()
    return escalation_id


if __name__ == "__main__":
    initialize_database()
    print("Database initialized.")
