"""
Stock screener — returns pre-filtered candidates for the CSP scanner.

Uses Yahoo Finance's built-in screener (yfinance.screen + EquityQuery) to narrow
the universe down to ~50-200 stocks before running the expensive options scan.

Filters applied:
  1. US region
  2. Price $MIN_PRICE – $MAX_PRICE  (within CSP strike range)
  3. Avg daily volume ≥ MIN_AVG_VOLUME  (proxy for "optionable + liquid")
  4. Beta range
  (MA200 filter is delegated to csp_scanner's ma200_score — asymmetric per cap.)

Usage:
    python screener.py                       # print tickers
    python screener.py --profile low         # conservative price/vol thresholds

    from screener import get_candidates
    tickers = get_candidates()               # returns list[str]
"""

from __future__ import annotations

import sys
from typing import Optional

import yfinance as yf
from yfinance.screener.query import EquityQuery

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_PRICE      = 60       # $ — avoid micro-cap / penny stocks
MAX_PRICE      = 300     # $ — screener pre-filter; csp_scanner MAX_STRIKE filters strike, not spot
MIN_AVG_VOLUME = 2500_000 # avg 3-month daily share volume (proxy for optionable)
MIN_BETA       = 0.3     # exclude very low beta (dull stocks, thin options)
MAX_BETA       = 4.0     # exclude very high beta (too volatile for CSP)
MAX_RESULTS    = 500     # max candidates returned (Yahoo limit per request ~250)
# MA200 filter intentionally removed — csp_scanner.ma200_score handles it
# asymmetrically per candidate (below MA200 → low score → drops in ranking).
# ─────────────────────────────────────────────────────────────────────────────


def _build_query() -> EquityQuery:
    """Build EquityQuery with numeric filters.
    MA200 check is done post-query since it's not a screener field."""
    # type: ignore comments below suppress yfinance stub imprecision —
    # EquityQuery operand types are correct at runtime but stubs are incomplete.
    return EquityQuery("and", [          # type: ignore[arg-type]
        EquityQuery("is-in", ["region", "us"]),
        EquityQuery("gt",    ["eodprice",     MIN_PRICE]),       # type: ignore[list-item]
        EquityQuery("lt",    ["eodprice",     MAX_PRICE]),       # type: ignore[list-item]
        EquityQuery("gt",    ["avgdailyvol3m", MIN_AVG_VOLUME]), # type: ignore[list-item]
        EquityQuery("gt",    ["beta",         MIN_BETA]),        # type: ignore[list-item]
        EquityQuery("lt",    ["beta",         MAX_BETA]),        # type: ignore[list-item]
    ])


def _fetch_all(query: EquityQuery, batch: int = 250) -> list[dict]:
    """Fetch all screener results, paginating if needed."""
    quotes: list[dict] = []
    offset = 0
    while True:
        r = yf.screen(query, offset=offset, count=batch)
        batch_quotes = r.get("quotes", [])
        quotes.extend(batch_quotes)
        total = r.get("total", 0)
        offset += len(batch_quotes)
        if offset >= total or not batch_quotes:
            break
    return quotes


def _post_filter(quotes: list[dict]) -> list[str]:
    """Extract symbols from screener quotes."""
    tickers = []
    for q in quotes:
        sym = q.get("symbol", "")
        if sym:
            tickers.append(sym.replace(".", "-"))
    return tickers


def get_candidates(
    min_price: float = MIN_PRICE,
    max_price: float = MAX_PRICE,
    min_avg_volume: int = MIN_AVG_VOLUME,
    verbose: bool = False,
) -> list[str]:
    """Return pre-filtered tickers suitable for short put scanning."""
    global MIN_PRICE, MAX_PRICE, MIN_AVG_VOLUME
    MIN_PRICE, MAX_PRICE, MIN_AVG_VOLUME = min_price, max_price, min_avg_volume

    query = _build_query()
    if verbose:
        print(f"Screening: price ${MIN_PRICE}–${MAX_PRICE}  "
              f"avg_vol≥{MIN_AVG_VOLUME:,}  beta {MIN_BETA}–{MAX_BETA}")

    quotes = _fetch_all(query)
    if verbose:
        print(f"  Raw results from Yahoo: {len(quotes)}")

    tickers = _post_filter(quotes)
    if verbose:
        print(f"  Candidates: {len(tickers)}")

    return tickers


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> dict:
    args = sys.argv[1:]
    opts: dict = {}
    i = 0
    while i < len(args):
        if args[i] == "--max-price" and i + 1 < len(args):
            opts["max_price"] = float(args[i + 1]); i += 2
        elif args[i] == "--min-vol" and i + 1 < len(args):
            opts["min_avg_volume"] = int(args[i + 1]); i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            if profile == "low":
                opts.update(max_price=150, min_avg_volume=1_000_000)
            elif profile == "high":
                opts.update(max_price=200, min_avg_volume=300_000)
            i += 2
        else:
            i += 1
    return opts


if __name__ == "__main__":
    opts = _parse_args()
    tickers = get_candidates(verbose=True, **opts)
    print(f"\n{len(tickers)} candidates:")
    print(", ".join(tickers))
