import sqlite3
from datetime import datetime, date
from contextlib import contextmanager
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_id INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                duration_seconds INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_table_date ON sessions(table_id, start_time)")
    print("База данных инициализирована")


def save_session(table_id, start_time, end_time):
    duration = int((end_time - start_time).total_seconds())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (table_id, start_time, end_time, duration_seconds) VALUES (?, ?, ?, ?)",
            (table_id, start_time, end_time, duration)
        )
    print(f"[DB] Стол {table_id}: записана сессия {duration} сек")


def get_stats_for_date(target_date=None):
    if target_date is None:
        target_date = date.today()
    
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT table_id, SUM(duration_seconds) as total_seconds, COUNT(*) as sessions_count
            FROM sessions
            WHERE DATE(start_time) = ?
            GROUP BY table_id
            ORDER BY table_id
        """, (target_date,)).fetchall()
    
    return [dict(row) for row in rows]


def get_stats_for_range(date_from, date_to):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT table_id, SUM(duration_seconds) as total_seconds, COUNT(*) as sessions_count
            FROM sessions
            WHERE DATE(start_time) BETWEEN ? AND ?
            GROUP BY table_id
            ORDER BY table_id
        """, (date_from, date_to)).fetchall()
    
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()