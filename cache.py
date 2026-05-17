"""SQLite disk cache for yfinance data with per-key TTL.

Keys follow the pattern  <type>:<symbol>[:<extra>], e.g.:
    fast_info:AAPL
    options:AAPL:2026-05-11
    history:AAPL
    calendar:AAPL
    sp500

Serialization: pickle (binary BLOB) — ~10x faster than JSON for DataFrames.
DB v2: yf_cache_v2.db  (old yf_cache.db with JSON encoding is no longer used)
"""

from __future__ import annotations

import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

DB_PATH = Path(__file__).parent / "yf_cache_v2.db"

# Seconds until a cached entry is considered stale
_TTL: dict[str, int] = {
    "sp500":       86400,  # 24 h — index composition rarely changes
    "russell2000": 86400,  # 24 h
    "history":     86400,  # 24 h — daily bars, one fetch per day is enough
    "calendar":    86400,  # 24 h — earnings dates don't move hourly
    "analyst":     86400,  # 24 h — ratings and price targets rarely change intraday
    "fast_info":   7200,   # 2 h
    "options":     7200,   # 2 h
}
_DEFAULT_TTL = 1200


def _ttl(key: str) -> int:
    prefix = key.split(":")[0]
    return _TTL.get(prefix, _DEFAULT_TTL)


# ── Serialization (pickle) ────────────────────────────────────────────────────

def _encode(value: Any) -> bytes:
    return pickle.dumps(value, protocol=4)


def _decode(data: bytes) -> Any:
    return pickle.loads(data)


# ── Cache class ───────────────────────────────────────────────────────────────

class Cache:
    def __init__(self, db_path: Path = DB_PATH):
        self._path = str(db_path)
        con = sqlite3.connect(self._path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, data BLOB NOT NULL, ts REAL NOT NULL)"
        )
        con.commit()
        con.close()

    def _con(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def get(self, key: str) -> Optional[Any]:
        con = self._con()
        row = con.execute(
            "SELECT data, ts FROM cache WHERE key=?", (key,)
        ).fetchone()
        con.close()
        if row is None:
            return None
        data, ts = row
        if time.time() - ts > _ttl(key):
            return None  # stale
        return _decode(data)

    def put(self, key: str, value: Any) -> None:
        con = self._con()
        con.execute(
            "INSERT OR REPLACE INTO cache (key, data, ts) VALUES (?,?,?)",
            (key, _encode(value), time.time()),
        )
        con.commit()
        con.close()

    def fetch(self, key: str, fn: Callable[[], Any]) -> Any:
        """Return cached value if fresh, otherwise call fn(), cache, and return."""
        v = self.get(key)
        if v is not None:
            return v
        v = fn()
        self.put(key, v)
        return v

    def clear(self, prefix: str | None = None) -> int:
        """Delete entries matching prefix (or all if None). Returns deleted count."""
        con = self._con()
        if prefix:
            cur = con.execute("DELETE FROM cache WHERE key LIKE ?", (f"{prefix}%",))
        else:
            cur = con.execute("DELETE FROM cache")
        con.commit()
        count = cur.rowcount
        con.close()
        return count

    def stats(self) -> dict[str, int]:
        """Row counts per key prefix."""
        con = self._con()
        rows = con.execute("SELECT key FROM cache").fetchall()
        con.close()
        counts: dict[str, int] = {}
        for (key,) in rows:
            prefix = key.split(":")[0]
            counts[prefix] = counts.get(prefix, 0) + 1
        return counts


# Module-level singleton used by the scanner
CACHE = Cache()
