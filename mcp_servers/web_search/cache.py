# mcp_servers/web_search/cache.py
import json
import time


class CacheLayer:
    def __init__(self, db_path: str = ".cache.db", ttl_seconds: int = 3600) -> None:
        import sqlite3

        self.ttl = ttl_seconds
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cache
               (key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"""
        )
        self.conn.commit()

    def get(self, key: str) -> list | None:
        row = self.conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row and time.time() < row[1]:
            return json.loads(row[0])
        return None

    def set(self, key: str, value: list) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time() + self.ttl),
        )
        self.conn.commit()

    def purge_expired(self) -> None:
        self.conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
        self.conn.commit()
