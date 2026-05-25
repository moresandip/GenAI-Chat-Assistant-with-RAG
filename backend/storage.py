import os
import sqlite3
import json
import logging
from typing import List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "rag_database.db")
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign keys support in SQLite
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        logger.info(f"Initializing SQLite database at: {self.db_path}")
        with self.get_connection() as conn:
            # 1. Users Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Documents Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Document Chunks Table (stores text & serialized embeddings)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,  -- JSON serialized float list
                    token_count INTEGER NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                );
            """)

            # 4. Chat Sessions Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                );
            """)

            # 5. Chat Messages Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,  -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    retrieved_chunks_count INTEGER DEFAULT 0,
                    similarity_scores TEXT DEFAULT '[]',  -- JSON serialized scores list
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
                );
            """)
            conn.commit()

    def clear_knowledge_base(self) -> None:
        """Clears all cached documents and embeddings."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM document_chunks;")
            conn.execute("DELETE FROM documents;")
            conn.commit()

    def add_document(self, title: str, content: str) -> int:
        """Inserts a document metadata record, returns its database ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO documents (title, content) VALUES (?, ?);",
                    (title, content)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # If document title exists, update its content instead
                cursor.execute(
                    "UPDATE documents SET content = ? WHERE title = ?;",
                    (content, title)
                )
                cursor.execute("SELECT id FROM documents WHERE title = ?;", (title,))
                row = cursor.fetchone()
                # Clear existing chunks for this document so we can re-chunk it
                if row:
                    doc_id = row[0]
                    cursor.execute("DELETE FROM document_chunks WHERE document_id = ?;", (doc_id,))
                    conn.commit()
                    return doc_id
                raise

    def add_chunk(self, document_id: int, chunk_index: int, text: str, embedding: List[float], token_count: int) -> None:
        """Stores a document chunk along with its vector embedding."""
        embedding_json = json.dumps(embedding)
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO document_chunks (document_id, chunk_index, text, embedding, token_count)
                VALUES (?, ?, ?, ?, ?);
                """,
                (document_id, chunk_index, text, embedding_json, token_count)
            )
            conn.commit()

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieves all chunks, joining document title metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.document_id, c.chunk_index, c.text, c.embedding, c.token_count, d.title as doc_title
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # --- User Auth Management ---

    def create_user(self, username: str, password_hash: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?);",
                (username, password_hash)
            )
            conn.commit()
            return cursor.lastrowid

    def get_user(self, username: str) -> Dict[str, Any] or None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?;", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- Chat Session & History Management ---

    def create_session(self, session_id: str, user_id: int, title: str) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?);",
                (session_id, user_id, title)
            )
            conn.commit()

    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC;",
                (user_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def session_exists(self, session_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM chat_sessions WHERE id = ?;", (session_id,))
            return cursor.fetchone() is not None

    def add_message(self, session_id: str, role: str, content: str, tokens_used: int = 0, retrieved_chunks_count: int = 0, similarity_scores: List[float] = None) -> None:
        if similarity_scores is None:
            similarity_scores = []
        scores_json = json.dumps(similarity_scores)
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, tokens_used, retrieved_chunks_count, similarity_scores)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (session_id, role, content, tokens_used, retrieved_chunks_count, scores_json)
            )
            conn.commit()

    def get_session_messages(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content, tokens_used, retrieved_chunks_count, similarity_scores, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?;
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                item = dict(row)
                item["similarity_scores"] = json.loads(item["similarity_scores"])
                result.append(item)
            return result
