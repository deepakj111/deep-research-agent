# mcp_servers/shared/cache.py
"""
SQLite-backed result cache shared by all MCP servers.

Keys are arbitrary strings (e.g. "web:{query}:{max_results}").
Values are JSON-serialisable lists (the raw normalised results list).
Expired rows are read-through filtered and lazily purged.
"""

import json
import sqlite3
import time
import typing


class CacheLayer:
    """SQLite-backed TTL cache for MCP tool responses with max-size limits."""

    def __init__(
        self, db_path: str = ".cache.db", ttl_seconds: int = 3600, max_entries: int = 10000
    ) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
        )
        self.conn.commit()

    def get(self, key: str) -> list[typing.Any] | None:
        row = self.conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row and time.time() < row[1]:
            return json.loads(row[0])
        return None

    def set(self, key: str, value: list[typing.Any]) -> None:
        self.purge_expired()

        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time() + self.ttl),
        )

        # Enforce max_entries limit by evicting the soonest-to-expire entries
        count = self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count > self.max_entries:
            to_delete = count - self.max_entries
            self.conn.execute(
                "DELETE FROM cache WHERE key IN (SELECT key FROM cache ORDER BY expires_at ASC LIMIT ?)",
                (to_delete,),
            )

        self.conn.commit()

    def purge_expired(self) -> None:
        self.conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()
