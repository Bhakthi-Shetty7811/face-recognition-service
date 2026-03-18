from psycopg2.pool import SimpleConnectionPool
import numpy as np
import logging
from .config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

logger = logging.getLogger(__name__)

class DB:

    def __init__(self):

        self.pool = SimpleConnectionPool(
            1,
            10,
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )

        self.create_tables()

        logger.info("PostgreSQL connection pool created")

    def create_tables(self):

        conn = self.pool.getconn()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS identities (
            id SERIAL PRIMARY KEY,
            name TEXT,
            embedding VECTOR(512)
        );
        """)

        conn.commit()

        cur.close()
        self.pool.putconn(conn)

    def add_identity(self, name, embedding):

        conn = self.pool.getconn()
        cur = conn.cursor()

        emb_list = embedding.tolist()

        cur.execute(
            "INSERT INTO identities (name, embedding) VALUES (%s, %s)",
            (name, emb_list)
        )

        conn.commit()

        cur.close()
        self.pool.putconn(conn)

        logger.info("Added identity %s", name)

    def list_identities(self):

        conn = self.pool.getconn()
        cur = conn.cursor()

        cur.execute("SELECT id, name FROM identities")

        rows = cur.fetchall()

        cur.close()
        self.pool.putconn(conn)

        return [{"id": r[0], "name": r[1]} for r in rows]

    def get_all_embeddings(self):

        conn = self.pool.getconn()
        cur = conn.cursor()

        cur.execute("SELECT name, embedding FROM identities")

        rows = cur.fetchall()

        names = []
        embs = []

        for name, embedding in rows:
            names.append(name)
            embs.append(np.array(embedding, dtype=np.float32))

        cur.close()
        self.pool.putconn(conn)

        return names, np.array(embs)