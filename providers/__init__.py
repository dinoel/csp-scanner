"""Provider factory — returns the configured DataProvider implementation."""
from __future__ import annotations

import os

from .base import DataProvider


def get_provider(name: str) -> DataProvider:
    """Instantiate and return the named DataProvider.

    Supported values:
        "yfinance" — 15-20 min delayed quotes via Yahoo Finance (no key needed)
        "massive"  — real-time quotes via Massive.com API (MASSIVE_API_KEY env var)
    """
    if name == "yfinance":
        from .yfinance_provider import YFinanceProvider
        return YFinanceProvider()

    if name == "massive":
        from .massive_provider import MassiveProvider
        api_key = os.environ.get("MASSIVE_API_KEY", "")
        return MassiveProvider(api_key=api_key)

    raise ValueError(
        f"Unknown DATA_PROVIDER {name!r} — choose: yfinance | massive"
    )
