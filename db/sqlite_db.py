# db/sqlite_db.py
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("app.db")


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT,
        rag_path TEXT,
        embedding_model TEXT,
        indexed_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_paper(paper_id, title, rag_path, embedding_model):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    INSERT OR REPLACE INTO papers
    VALUES (?, ?, ?, ?, ?)
    """, (
        paper_id,
        title,
        rag_path,
        embedding_model,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()


def get_paper(paper_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT title, rag_path, embedding_model
    FROM papers
    WHERE paper_id = ?
    """, (paper_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "title": row[0],
        "rag_path": row[1],
        "embedding_model": row[2]
    }
