"""Massive.com data provider (formerly Polygon.io) — real-time quotes.

Requires:
    pip install requests
    MASSIVE_API_KEY env var set to your Massive.com API key

API docs: https://massive.com/docs/rest/quickstart

Quick diagnosis — run from project root:
    python -m providers.massive_provider AAPL
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .base import DataProvider

_BASE = "https://api.massive.com"


class MassiveProvider(DataProvider):
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "MASSIVE_API_KEY env var is not set — required for MassiveProvider"
            )
        self._key = api_key
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, **params: Any) -> Any:
        resp = self._session.get(f"{_BASE}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, path: str, **params: Any) -> list:
        """Fetch all pages of a results array, following next_url."""
        results: list = []
        url: str | None = f"{_BASE}{path}"
        p = dict(params)
        while url:
            resp = self._session.get(url, params=p, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results") or [])
            url = data.get("next_url")
            p = {}  # next_url already has all params encoded
        return results

    # ── DataProvider interface ────────────────────────────────────────────────

    def get_spot_and_expirations(self, symbol: str) -> dict:
        # Current price — let exceptions propagate so the caller can log them
        snap = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        t = snap.get("ticker", {})
        price = float(
            t.get("lastTrade", {}).get("p")
            or t.get("day", {}).get("c")
            or t.get("prevDay", {}).get("c")
            or 0.0
        )

        # Available expiration dates — "false" must be a lowercase string,
        # not Python's False (requests would send "False" and the API ignores it)
        try:
            contracts = self._paginate(
                "/v3/reference/options/contracts",
                underlying_ticker=symbol,
                expired="false",
                contract_type="put",  # only puts needed for DTE discovery
                limit=1000,
            )
            expirations = sorted(
                {c["expiration_date"] for c in contracts if c.get("expiration_date")}
            )
        except Exception as e:
            print(f"[{symbol}] Massive: failed to fetch expirations: {e!r}")
            expirations = []

        return {"price": price, "options": expirations}

    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        results = self._paginate(
            f"/v3/snapshot/options/{symbol}",
            expiration_date=expiry,
            limit=250,
        )
        calls: list[dict] = []
        puts:  list[dict] = []
        for r in results:
            details       = r.get("details", {})
            contract_type = details.get("contract_type", "")
            quote         = r.get("last_quote", {})
            day           = r.get("day", {})
            last_trade    = r.get("last_trade", {})
            row = {
                "strike":       details.get("strike_price", float("nan")),
                "bid":          float(quote.get("bid") or 0),
                "ask":          float(quote.get("ask") or 0),
                "lastPrice":    float(last_trade.get("price") or r.get("fmv") or 0),
                "volume":       int(day.get("volume") or 0),
                "openInterest": int(r.get("open_interest") or 0),
            }
            if contract_type == "call":
                calls.append(row)
            elif contract_type == "put":
                puts.append(row)

        return {
            "calls": pd.DataFrame(calls) if calls else pd.DataFrame(),
            "puts":  pd.DataFrame(puts)  if puts  else pd.DataFrame(),
        }

    def get_history(self, symbol: str) -> pd.DataFrame:
        end   = datetime.now()
        start = end - timedelta(days=400)  # slightly more than 1y to ensure 252+ trading days
        data  = self._get(
            f"/v2/aggs/ticker/{symbol}/range/1/day"
            f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            adjusted="true",
            sort="asc",
            limit=500,
        )
        results = data.get("results") or []
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        # Polygon/Massive aggs: o=open h=high l=low c=close v=volume t=timestamp_ms
        df = df.rename(columns={"c": "Close", "o": "Open", "h": "High",
                                 "l": "Low", "v": "Volume"})
        df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
        return df.set_index("Date").sort_index()

    def get_calendar(self, symbol: str) -> dict:
        try:
            data   = self._get(f"/vX/reference/tickers/{symbol}/events")
            events = (data.get("results") or {}).get("events", [])
            dates  = [e["date"] for e in events
                      if e.get("type") == "earnings" and e.get("date")]
            if dates:
                return {"Earnings Date": dates}
        except Exception:
            pass
        return {}

    def get_analyst_info(self, symbol: str) -> dict:
        # Massive/Polygon doesn't provide analyst consensus — fall back to yfinance
        import yfinance as yf
        info = yf.Ticker(symbol).info
        target_raw = info.get("targetMeanPrice")
        return {
            "rating":       info.get("recommendationKey") or "",
            "target":       float(target_raw) if target_raw else float("nan"),
            "num_analysts": int(info.get("numberOfAnalystOpinions") or 0),
        }

    # ── Diagnostic ───────────────────────────────────────────────────────────

    def diagnose(self, symbol: str) -> None:
        """Print what each API call returns for a single ticker. Run to debug."""
        print(f"\n=== Massive provider diagnostic for {symbol} ===\n")

        print("1. Stocks snapshot (price)...")
        try:
            snap = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
            t = snap.get("ticker", {})
            price = (t.get("lastTrade", {}).get("p")
                     or t.get("day", {}).get("c")
                     or t.get("prevDay", {}).get("c"))
            print(f"   price = {price}")
            print(f"   raw ticker keys: {list(t.keys())}")
        except Exception as e:
            print(f"   FAILED: {e!r}")

        print("\n2. Options contracts reference (expirations)...")
        try:
            resp = self._session.get(
                f"{_BASE}/v3/reference/options/contracts",
                params={"underlying_ticker": symbol, "expired": "false",
                        "contract_type": "put", "limit": 10},
                timeout=15,
            )
            print(f"   HTTP {resp.status_code}")
            data = resp.json()
            results = data.get("results") or []
            print(f"   results count (first page, limit=10): {len(results)}")
            if results:
                print(f"   sample expiration_date: {results[0].get('expiration_date')}")
        except Exception as e:
            print(f"   FAILED: {e!r}")

        print("\n3. Options snapshot (chain, first available expiry)...")
        try:
            resp = self._session.get(
                f"{_BASE}/v3/snapshot/options/{symbol}",
                params={"limit": 5},
                timeout=15,
            )
            print(f"   HTTP {resp.status_code}")
            data = resp.json()
            results = data.get("results") or []
            print(f"   results count (limit=5): {len(results)}")
            if results:
                r = results[0]
                print(f"   sample: contract_type={r.get('details',{}).get('contract_type')} "
                      f"strike={r.get('details',{}).get('strike_price')} "
                      f"bid={r.get('last_quote',{}).get('bid')} "
                      f"ask={r.get('last_quote',{}).get('ask')} "
                      f"OI={r.get('open_interest')} "
                      f"vol={r.get('day',{}).get('volume')}")
        except Exception as e:
            print(f"   FAILED: {e!r}")

        print("\n4. Historical bars (price history)...")
        try:
            end   = datetime.now()
            start = end - timedelta(days=30)
            resp  = self._session.get(
                f"{_BASE}/v2/aggs/ticker/{symbol}/range/1/day"
                f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
                params={"adjusted": "true", "sort": "asc", "limit": 5},
                timeout=15,
            )
            print(f"   HTTP {resp.status_code}")
            data    = resp.json()
            results = data.get("results") or []
            print(f"   bars returned (limit=5): {len(results)}")
            if results:
                print(f"   last bar close: {results[-1].get('c')}")
        except Exception as e:
            print(f"   FAILED: {e!r}")

        print()


if __name__ == "__main__":
    import os, sys
    api_key = os.environ.get("MASSIVE_API_KEY", "")
    symbol  = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    MassiveProvider(api_key=api_key).diagnose(symbol)
