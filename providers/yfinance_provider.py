"""yfinance data provider — 15-20 min delayed quotes."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from .base import DataProvider


class YFinanceProvider(DataProvider):
    def get_spot_and_expirations(self, symbol: str) -> dict:
        tk = yf.Ticker(symbol)
        try:
            price = float(tk.history(period="5d", auto_adjust=True)["Close"].iloc[-1])
        except Exception:
            price = 0.0
        return {"price": price, "options": list(tk.options or [])}

    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        chain = yf.Ticker(symbol).option_chain(expiry)
        return {"calls": chain.calls, "puts": chain.puts}

    def get_history(self, symbol: str) -> pd.DataFrame:
        return yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)

    def get_calendar(self, symbol: str) -> dict:
        cal = yf.Ticker(symbol).calendar
        if isinstance(cal, pd.DataFrame):
            return cal.to_dict()
        return cal or {}

    def get_analyst_info(self, symbol: str) -> dict:
        info = yf.Ticker(symbol).info
        target_raw = info.get("targetMeanPrice")
        return {
            "rating":       info.get("recommendationKey") or "",
            "target":       float(target_raw) if target_raw else float("nan"),
            "num_analysts": int(info.get("numberOfAnalystOpinions") or 0),
        }
