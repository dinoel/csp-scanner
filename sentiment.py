"""StockTwits sentiment for a ticker symbol.

Requires a free StockTwits API token:
  1. Register at https://stocktwits.com/developers/apps
  2. Create an app → copy the OAuth token
  3. Set env var: export STOCKTWITS_TOKEN=your_token

Returns bull/bear % based on tagged messages (last ~30 posts).
Only messages where the author explicitly tagged Bullish/Bearish are counted;
untagged posts are ignored so the ratio reflects conviction, not volume.

Without a token: all calls return NaN gracefully — scanner still works.
"""
from __future__ import annotations

import math
import os

import requests

_BASE    = "https://api.stocktwits.com/api/2"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

_NAN = float("nan")
_EMPTY = {"bull_pct": _NAN, "bear_pct": _NAN, "tagged_count": 0}

# Module-level token — set once via init() or STOCKTWITS_TOKEN env var
TOKEN: str = ""


def init(token: str | None = None) -> None:
    """Set the API token. Call once at startup (reads env var if not passed)."""
    global TOKEN
    TOKEN = token or os.environ.get("STOCKTWITS_TOKEN", "")


def fetch_sentiment(symbol: str) -> dict:
    """Fetch bull/bear sentiment from the last ~30 StockTwits messages.

    Returns:
        {
            "bull_pct":     float  # % bullish of tagged messages (NaN if no data)
            "bear_pct":     float  # % bearish
            "tagged_count": int    # messages with explicit sentiment tag
        }
    """
    if not TOKEN:
        return _EMPTY

    try:
        resp = _SESSION.get(
            f"{_BASE}/streams/symbol/{symbol}.json",
            params={"access_token": TOKEN},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[{symbol}] StockTwits: {e!r}")
        return _EMPTY

    messages = data.get("messages") or []
    bull = bear = 0
    for m in messages:
        basic = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        if basic == "Bullish":
            bull += 1
        elif basic == "Bearish":
            bear += 1

    tagged = bull + bear
    if tagged == 0:
        return _EMPTY

    bull_pct = round(bull / tagged * 100, 1)
    return {
        "bull_pct":     bull_pct,
        "bear_pct":     round(100.0 - bull_pct, 1),
        "tagged_count": tagged,
    }
