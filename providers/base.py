"""Abstract DataProvider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def get_spot_and_expirations(self, symbol: str) -> dict:
        """Return {"price": float, "options": list[str ISO-format dates]}."""

    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: str) -> dict:
        """Return {"calls": DataFrame, "puts": DataFrame}.

        Required DataFrame columns:
            strike, bid, ask, lastPrice, volume, openInterest
        """

    @abstractmethod
    def get_history(self, symbol: str) -> pd.DataFrame:
        """Return 252+ days of daily OHLCV. Must have a 'Close' column."""

    @abstractmethod
    def get_calendar(self, symbol: str) -> dict:
        """Return {"Earnings Date": list[date | str]} or empty dict."""

    @abstractmethod
    def get_analyst_info(self, symbol: str) -> dict:
        """Return {"rating": str, "target": float, "num_analysts": int}."""
