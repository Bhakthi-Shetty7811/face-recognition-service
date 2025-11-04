import sqlite3
import numpy as np
import os

class DB:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                embedding BLOB
            )
        """)
        self.conn.commit()

    def add_identity(self, name, embedding):
        emb_bytes = embedding.astype(np.float32).tobytes()
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        self.conn.execute("INSERT INTO identities (name, embedding) VALUES (?, ?)", (name, emb_bytes))
        self.conn.commit()

    def list_identities(self):
        cur = self.conn.execute("SELECT id, name FROM identities")
        return [{"id": row[0], "name": row[1]} for row in cur.fetchall()]

    def get_all_embeddings(self):
        cur = self.conn.execute("SELECT name, embedding FROM identities")
        names, embs = [], []
        for row in cur.fetchall():
            names.append(row[0])
            embs.append(np.frombuffer(row[1], dtype=np.float32))
        return names, np.array(embs)
