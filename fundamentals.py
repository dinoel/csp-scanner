"""Fundamental analysis via financialdatasets.ai API.

Computes a 0-100 quality score from four metrics that matter for CSP sellers:

  eps_beat_rate  (35%) — how often does the company beat EPS estimates?
                         High beat rate → less earnings-gap risk
  rev_growth_pct (25%) — YoY revenue growth signals business momentum
  fcf_margin_pct (25%) — free cash flow / revenue; financial health proxy
  debt_to_equity (15%) — lower leverage → less tail risk

Requires: FINANCIALDATASETS_API_KEY env var.
Cache TTL: 24 h (quarterly reports don't change intraday).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import requests

_BASE = "https://api.financialdatasets.ai"

_NAN = float("nan")


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class FundamentalMetrics:
    rev_growth_pct: float   # YoY revenue growth %
    eps_beat_rate:  float   # % of last 8 quarters where actual >= estimated (0-100)
    fcf_margin_pct: float   # FCF / revenue * 100 (latest annual)
    debt_to_equity: float   # total debt / total equity
    score:          float   # composite 0-100


_EMPTY = FundamentalMetrics(
    rev_growth_pct=_NAN, eps_beat_rate=_NAN,
    fcf_margin_pct=_NAN, debt_to_equity=_NAN,
    score=_NAN,
)


# ── API client ────────────────────────────────────────────────────────────────

class FundamentalsClient:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("FINANCIALDATASETS_API_KEY is not set")
        self._session = requests.Session()
        self._session.headers.update({"X-API-KEY": api_key})

    def _get(self, path: str, **params) -> dict:
        resp = self._session.get(f"{_BASE}{path}", params=params, timeout=15)
        if resp.status_code == 404:
            return {}   # no data for this ticker — not an error
        resp.raise_for_status()
        return resp.json()

    # ── raw fetchers (called by CACHE.fetch as lambdas) ───────────────────────

    def fetch_income(self, symbol: str) -> list[dict]:
        data = self._get("/financials/income-statements",
                         ticker=symbol, period="annual", limit=3)
        return data.get("income_statements") or []

    def fetch_cashflow(self, symbol: str) -> list[dict]:
        data = self._get("/financials/cash-flow-statements",
                         ticker=symbol, period="annual", limit=2)
        return data.get("cash_flow_statements") or []

    def fetch_balance(self, symbol: str) -> list[dict]:
        data = self._get("/financials/balance-sheets",
                         ticker=symbol, period="annual", limit=2)
        return data.get("balance_sheets") or []

    def fetch_earnings(self, symbol: str) -> list[dict]:
        data = self._get("/earnings", ticker=symbol, limit=8)
        return data.get("earnings") or []

    def fetch_news(self, symbol: str) -> list[dict]:
        data = self._get("/news", ticker=symbol, limit=5)
        return data.get("news") or []


# ── Module-level singleton (set by csp_scanner on startup) ───────────────────

CLIENT: FundamentalsClient | None = None


def init(api_key: str | None = None) -> None:
    """Create the module-level CLIENT. Call once at startup."""
    global CLIENT
    key = api_key or os.environ.get("FINANCIALDATASETS_API_KEY", "")
    if key:
        CLIENT = FundamentalsClient(key)


# ── Metric extractors ─────────────────────────────────────────────────────────

def _f(d: dict, *keys) -> float:
    """Safe float extraction with fallback keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return _NAN


def _rev_growth(income: list[dict]) -> float:
    if len(income) < 2:
        return _NAN
    curr = _f(income[0], "revenue", "total_revenue")
    prev = _f(income[1], "revenue", "total_revenue")
    if math.isnan(curr) or math.isnan(prev) or prev == 0:
        return _NAN
    return (curr - prev) / abs(prev) * 100.0


def _fcf_margin(income: list[dict], cashflow: list[dict]) -> float:
    if not cashflow or not income:
        return _NAN
    fcf = _f(cashflow[0], "free_cash_flow", "free_cash_flow_firm")
    if math.isnan(fcf):
        op  = _f(cashflow[0], "operating_cash_flow", "cash_from_operations")
        cap = _f(cashflow[0], "capital_expenditure", "capex", "purchase_of_property_plant_and_equipment")
        if math.isnan(op) or math.isnan(cap):
            return _NAN
        fcf = op + cap  # capex is stored as negative in most APIs
    rev = _f(income[0], "revenue", "total_revenue")
    if math.isnan(rev) or rev == 0:
        return _NAN
    return fcf / rev * 100.0


def _debt_to_equity(balance: list[dict]) -> float:
    if not balance:
        return _NAN
    bs   = balance[0]
    debt = _f(bs, "total_debt", "long_term_debt", "total_long_term_debt")
    eq   = _f(bs, "total_equity", "shareholders_equity",
               "total_stockholders_equity", "total_equity_gross_minority_interest")
    if math.isnan(debt) or math.isnan(eq) or eq == 0:
        return _NAN
    return debt / abs(eq)


def _beat_rate(earnings: list[dict]) -> float:
    valid = [
        e for e in earnings
        if e.get("actual_eps") is not None and e.get("estimated_eps") is not None
    ]
    if not valid:
        return _NAN
    beats = sum(
        1 for e in valid
        if float(e["actual_eps"]) >= float(e["estimated_eps"])
    )
    return beats / len(valid) * 100.0


# ── Sub-scores (each 0-100) ───────────────────────────────────────────────────

def _score_rev_growth(pct: float) -> float:
    if math.isnan(pct): return 50.0
    if pct >= 25: return 95
    if pct >= 15: return 80
    if pct >=  5: return 65
    if pct >=  0: return 45
    if pct >= -10: return 25
    return 5.0


def _score_beat_rate(rate_pct: float) -> float:
    if math.isnan(rate_pct): return 50.0
    if rate_pct >= 87: return 95
    if rate_pct >= 75: return 80
    if rate_pct >= 62: return 60
    if rate_pct >= 50: return 35
    return 10.0


def _score_fcf_margin(pct: float) -> float:
    if math.isnan(pct): return 50.0
    if pct >= 20: return 95
    if pct >= 10: return 75
    if pct >=  5: return 55
    if pct >=  0: return 30
    return 5.0


def _score_debt_equity(ratio: float) -> float:
    if math.isnan(ratio): return 50.0
    if ratio < 0:   return 90.0   # net cash position
    if ratio < 0.5: return 85
    if ratio < 1.0: return 70
    if ratio < 2.0: return 50
    if ratio < 3.0: return 25
    return 5.0


def _composite(rg: float, br: float, fcf: float, de: float) -> float:
    return (0.35 * _score_beat_rate(br)   +
            0.25 * _score_rev_growth(rg)  +
            0.25 * _score_fcf_margin(fcf) +
            0.15 * _score_debt_equity(de))


# ── Public API ────────────────────────────────────────────────────────────────

def compute(income: list[dict], cashflow: list[dict],
            balance: list[dict], earnings: list[dict]) -> FundamentalMetrics:
    """Build FundamentalMetrics from raw API lists."""
    rg  = _rev_growth(income)
    br  = _beat_rate(earnings)
    fcf = _fcf_margin(income, cashflow)
    de  = _debt_to_equity(balance)
    sc  = _composite(rg, br, fcf, de)
    return FundamentalMetrics(
        rev_growth_pct = round(rg,  1) if not math.isnan(rg)  else _NAN,
        eps_beat_rate  = round(br,  1) if not math.isnan(br)  else _NAN,
        fcf_margin_pct = round(fcf, 1) if not math.isnan(fcf) else _NAN,
        debt_to_equity = round(de,  2) if not math.isnan(de)  else _NAN,
        score          = round(sc,  1),
    )


def fetch_and_compute(symbol: str, cache) -> FundamentalMetrics:
    """Fetch all four data sources (cached) and return FundamentalMetrics.

    Uses the module-level CLIENT. Returns _EMPTY if CLIENT is not initialised
    or if the API call fails — so the scanner degrades gracefully.
    """
    if CLIENT is None:
        return _EMPTY
    try:
        income   = cache.fetch(f"fd_income:{symbol}",   lambda: CLIENT.fetch_income(symbol))
        cashflow = cache.fetch(f"fd_cashflow:{symbol}", lambda: CLIENT.fetch_cashflow(symbol))
        balance  = cache.fetch(f"fd_balance:{symbol}",  lambda: CLIENT.fetch_balance(symbol))
        earnings = cache.fetch(f"fd_earnings:{symbol}", lambda: CLIENT.fetch_earnings(symbol))
        return compute(income, cashflow, balance, earnings)
    except Exception as e:
        print(f"[{symbol}] fundamentals error: {e!r}")
        return _EMPTY
