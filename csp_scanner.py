"""
Short Put (Cash-Secured Put) scanner for US equity options.

For each ticker in the universe, finds the put option within DTE_MIN-DTE_MAX
closest to TARGET_DELTA and reports premium, break-even, return, and
probability of profit. Warns when earnings fall within the expiry window.

Data providers (set DATA_PROVIDER in config):
    yfinance — 15-20 min delayed quotes, no API key needed (default)
    massive  — real-time quotes via Massive.com, set MASSIVE_API_KEY env var

Usage:
    python csp_scanner.py
todo: filter based on volume/spread , use mid price instead of bid
"""

from __future__ import annotations

import html as _he
import io
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
# LOG_LEVEL env var controls verbosity. Defaults to INFO (per-ticker outcomes
# and progress). Set LOG_LEVEL=DEBUG for per-strike rejection reasons,
# WARNING to suppress everything except errors/rate-limits.
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(message)s",
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    stream=sys.stdout,
    force=True,  # override prior basicConfig from libraries
)
log = logging.getLogger("csp")

# Human-readable labels for the rejection counters produced by find_best_put.
REJECT_LABELS = {
    "oi":         "OI < MIN_OPEN_INTEREST",
    "vol":        "vol < MIN_VOLUME",
    "stale":      "stale lastTradeDate",
    "spread":     "spread > MAX_SPREAD_PCT",
    "no_quote":   "no bid/ask/last",
    "iv_solve":   "IV solver failed",
    "delta_out":  "|delta| out of range",
    "strike_max": "strike > MAX_STRIKE",
}

import analytics
import fundamentals as _fund
import sentiment as _sentiment
import skew as skewlib
import strategies as _strategies
from ai import ai_analysis
from cache import CACHE
from models import PutRow
from providers import get_provider
from report import RATING_ABBREV, build_profile_block, write_scan_bundle

# ── CONFIG ────────────────────────────────────────────────────────────────────
#UNIVERSE: str | list[str] = ["NVDA", "INTC", "AMD", "PLTR", "MCD", "ASTS", "TEAM", "NBIS", "IREN", "EMR", "ORCL", "DELL", "SMCI", "HAL", "GLW", "FCX", "HUT", "ARM"]
#UNIVERSE: str | list[str] = ["CRWV", "NVDA", "INTC", "MCD", "ASTS"]
UNIVERSE: str | list[str] = "screener"
SAMPLE_SIZE: int | None = None        # tickers to sample; None = full universe
RISK_PROFILE = "medium"              # "low" | "medium" | "high"
STRATEGY     = os.getenv("STRATEGY", "bps").lower()  # "csp" | "bps"  (env-overridable)
BPS_MAX_WIDTH = 20.0   # widest bull put spread (long-leg distance, $) considered
BPS_MIN_CREDIT = 0.05  # require at least this much credit per share
RISK_FREE_RATE = 0.05
MIN_BID = 0.05
COMPUTE_IV_RANK = True
MAX_WORKERS = 6      # parallel ticker scans; keep ≤8 to avoid Yahoo rate limits
RESULTS_DIR = "results"
PROFILES_TO_RUN = ["low", "medium", "high"]
#PROFILES_TO_RUN = ["medium"]
DEFAULT_PROFILE = "medium"  # which profile the index links to by default
def _envbool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "y")

ENABLE_AI_ANALYSIS  = _envbool("ENABLE_AI_ANALYSIS",  False)
ENABLE_FUNDAMENTALS = _envbool("ENABLE_FUNDAMENTALS", False)
NON_INTERACTIVE     = bool(os.getenv("CI"))  # GitHub Actions sets CI=true
AI_TOP_N = 10
AI_BATCH_SIZE = 10   # hard cap per batch; prompts before fetching the next batch
AI_MODEL = "claude-opus-4-7"
DATA_PROVIDER = "yfinance"  # "yfinance" | "massive" (set MASSIVE_API_KEY env var)
MARKET_OPEN_GRACE_MIN = 30  # minutes after market open before requiring today's volume

# ── Risk profiles ─────────────────────────────────────────────────────────────
_PROFILES: dict[str, dict] = {
    "low": dict(
        DTE_MIN=21,  DTE_MAX=51,
        DELTA_MIN=0.05, DELTA_MAX=0.20,
        MAX_STRIKE=150,
        MIN_OPEN_INTEREST=100, MIN_VOLUME=25, MAX_SPREAD_PCT=0.30,
        SCORE_W_PROB=0.50, SCORE_W_RETURN=0.13, SCORE_W_SAFETY=0.18,
        SCORE_W_MA200=0.09, SCORE_W_FUNDAMENTAL=0.10,
    ),
    "medium": dict(
        DTE_MIN=7,   DTE_MAX=51,
        DELTA_MIN=0.05, DELTA_MAX=0.50,
        MAX_STRIKE=200,
        MIN_OPEN_INTEREST=50,  MIN_VOLUME=200,  MAX_SPREAD_PCT=0.50,
        SCORE_W_PROB=0.40, SCORE_W_RETURN=0.22, SCORE_W_SAFETY=0.18,
        SCORE_W_MA200=0.08, SCORE_W_FUNDAMENTAL=0.12,
    ),
    "high": dict(
        DTE_MIN=7,   DTE_MAX=31,
        DELTA_MIN=0.15, DELTA_MAX=0.50,
        MAX_STRIKE=200,
        MIN_OPEN_INTEREST=50,  MIN_VOLUME=200,  MAX_SPREAD_PCT=0.30,
        SCORE_W_PROB=0.27, SCORE_W_RETURN=0.40, SCORE_W_SAFETY=0.13,
        SCORE_W_MA200=0.08, SCORE_W_FUNDAMENTAL=0.12,
    ),
}

if RISK_PROFILE not in _PROFILES:
    raise ValueError(f"Unknown RISK_PROFILE {RISK_PROFILE!r} — choose: {list(_PROFILES)}")
globals().update(_PROFILES[RISK_PROFILE])

PROVIDER = get_provider(DATA_PROVIDER)
_fund.init()      # reads FINANCIALDATASETS_API_KEY from env; no-op if key absent
_sentiment.init() # reads STOCKTWITS_TOKEN from env; no-op if key absent
# ─────────────────────────────────────────────────────────────────────────────

SP500_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
    "/refs/heads/main/data/constituents.csv"
)
IWM_URL = (
    "https://www.ishares.com/us/products/239710/x"
    "/1467271812596.ajax?fileType=csv&dataType=fund"
)


# ── Utility helpers ───────────────────────────────────────────────────────────

def _int(v) -> int:
    """int() safe for NaN/None."""
    try:
        f = float(v)
        return 0 if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


def _r(x, n: int = 2) -> float:
    """round() safe for NaN."""
    try:
        f = float(x)
        return round(f, n) if not math.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _ev_binary(p_pct: float, gain: float, loss: float) -> float:
    """Binary expected value: P × max_gain − (1 − P) × max_loss. NaN-safe."""
    if any(math.isnan(v) for v in (p_pct, gain, loss)):
        return float("nan")
    p = p_pct / 100.0
    return p * gain - (1.0 - p) * loss


# ── Trade-date staleness ─────────────────────────────────────────────────────

def _last_valid_trade_date(_now: datetime | None = None) -> date:
    """Return the most recent date we'd accept option volume from.

    Rules (all times in US/Eastern):
      - Weekend (Sat/Sun)          → previous Friday
      - Weekday, before market open + grace period (< 9:30 + MARKET_OPEN_GRACE_MIN)
                                   → previous business day
      - Weekday, within grace (9:30 … 9:30+grace)
                                   → previous business day
      - Weekday, after grace       → today

    Pass `_now` to override the current time (used in tests).
    """
    from zoneinfo import ZoneInfo
    from datetime import timedelta as _td

    et = ZoneInfo("America/New_York")
    now_et = (_now.astimezone(et) if _now is not None else datetime.now(et))
    today  = now_et.date()
    wd     = today.weekday()   # 0=Mon … 6=Sun

    def _prev_biz(d: date) -> date:
        d -= _td(days=1)
        while d.weekday() >= 5:   # skip Sat/Sun
            d -= _td(days=1)
        return d

    if wd >= 5:   # Saturday or Sunday → Friday
        return _prev_biz(today)

    # Weekday: check if we're past the grace window
    market_open  = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    grace_end    = market_open + __import__("datetime").timedelta(minutes=MARKET_OPEN_GRACE_MIN)
    if now_et >= grace_end:
        return today        # past grace — require today's trades
    return _prev_biz(today) # pre-open or within grace — accept prev business day


# ── Scoring (depends on SCORE_W_* config globals) ────────────────────────────

def score_put(ann_rtn: float, profit_prob: float, vs_em: float,
              ma200_sc: float = 50.0, fund_sc: float = 50.0) -> float:
    """Composite score for a (strike, expiry) candidate. Higher = better."""
    if math.isnan(ann_rtn) or math.isnan(profit_prob):
        return float("-inf")
    exp_rtn = min(ann_rtn * profit_prob / 100.0, 200.0)
    em_buf  = min(max(0.0 if math.isnan(vs_em) else vs_em, 0.0), 150.0)
    fund_sc = 50.0 if math.isnan(fund_sc) else fund_sc
    return (SCORE_W_RETURN      * exp_rtn +
            SCORE_W_PROB        * profit_prob +
            SCORE_W_SAFETY      * em_buf +
            SCORE_W_MA200       * ma200_sc +
            SCORE_W_FUNDAMENTAL * fund_sc)


# ── Liquidity filter (depends on MIN_* config globals) ───────────────────────

def _effective_price(row, oi_unavailable: bool = False) -> tuple[float, float, str]:
    """Return (price_for_iv, bid_for_return, reject_reason).

    `reject_reason` is "" on success, otherwise a REJECT_LABELS key naming
    which liquidity stage filtered the contract out.

    If `oi_unavailable` is True (Yahoo returned openInterest=0 for the entire
    chain — known quirk), the OI filter is skipped for this row.
    """
    if not oi_unavailable:
        oi = _int(row.get("openInterest"))
        if oi < MIN_OPEN_INTEREST:
            return float("nan"), float("nan"), "oi"

    vol = _int(row.get("volume"))
    if vol < MIN_VOLUME:
        return float("nan"), float("nan"), "vol"

    # Staleness check: yfinance 'volume' is from the last session the contract
    # actually traded — not necessarily today. Reject options whose last trade
    # predates the most recent valid trading session.
    last_trade = row.get("lastTradeDate")
    if last_trade is not None:
        try:
            trade_date = (last_trade.date() if hasattr(last_trade, "date")
                          else datetime.strptime(str(last_trade)[:10], "%Y-%m-%d").date())
            if trade_date < _last_valid_trade_date():
                return float("nan"), float("nan"), "stale"
        except Exception:
            pass

    bid  = float(row.get("bid")       or 0)
    ask  = float(row.get("ask")       or 0)
    last = float(row.get("lastPrice") or 0)

    if bid > 0 and ask > 0:
        if (ask - bid) / bid > MAX_SPREAD_PCT:
            return float("nan"), float("nan"), "spread"
        return (bid + ask) / 2, bid, ""

    if bid > MIN_BID:
        return bid, bid, ""

    if last > MIN_BID:
        if vol < max(MIN_VOLUME * 5, 50):
            return float("nan"), float("nan"), "stale"
        return last, last, ""

    return float("nan"), float("nan"), "no_quote"


# ── Scan helpers ──────────────────────────────────────────────────────────────

pick_expiry = analytics.pick_expiry   # re-export so existing imports still work


def check_earnings(cal: dict, today, expiry_date) -> Optional[str]:
    """Return earnings date string 'YYYY-MM-DD' if earnings fall in window, else None."""
    try:
        if not cal:
            return None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        elif isinstance(cal, pd.DataFrame):
            dates = cal.get("Earnings Date", pd.Series()).tolist()
        else:
            return None
        if not isinstance(dates, (list, tuple)):
            dates = [dates]
        for ed in dates:
            if hasattr(ed, "date"):
                ed = ed.date()
            elif isinstance(ed, str):
                try:
                    ed = datetime.strptime(ed[:10], "%Y-%m-%d").date()
                except ValueError:
                    continue
            if today <= ed <= expiry_date:
                return ed.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def find_best_put(puts: pd.DataFrame, spot: float, T: float, r: float,
                  ma200: float, em: float) -> tuple[Optional[tuple], Counter]:
    """Score every eligible put strike and return (best, stats).

    `stats` is a Counter with keys: "examined", "considered" (passed all
    filters), and one entry per REJECT_LABELS key counting which stage
    rejected each strike.
    """
    best = None
    best_score = float("-inf")
    dte = max(T * 365.0, 1.0)
    stats: Counter = Counter()

    # Yahoo quirk: sometimes the API returns openInterest=0 for an entire
    # chain (whole-market, not per-contract). When that's the case, skip the
    # OI filter for this chain — otherwise every strike would be wrongly
    # rejected. The chain-level fact is tracked in stats["oi_unavailable"].
    oi_unavailable = (
        "openInterest" in puts.columns
        and len(puts) > 0
        and float(puts["openInterest"].fillna(0).max() or 0) == 0
    )
    if oi_unavailable:
        stats["oi_unavailable"] += 1

    _need = [c for c in ("strike", "bid", "ask", "lastPrice", "volume",
                         "openInterest", "lastTradeDate")
             if c in puts.columns]
    for row in puts[_need].to_dict("records"):
        stats["examined"] += 1
        price_iv, bid_ret, reason = _effective_price(row, oi_unavailable=oi_unavailable)
        if reason:
            stats[reason] += 1
            continue
        strike = float(row["strike"])
        iv = analytics.compute_iv(spot, strike, T, r, price_iv)
        if math.isnan(iv):
            stats["iv_solve"] += 1
            continue
        delta = analytics.bs_put_delta(spot, strike, T, r, iv)
        if math.isnan(delta) or not (DELTA_MIN <= abs(delta) <= DELTA_MAX):
            stats["delta_out"] += 1
            continue
        if MAX_STRIKE and strike > MAX_STRIKE:
            stats["strike_max"] += 1
            continue

        stats["considered"] += 1
        pp    = analytics.calc_profit_prob(spot, strike, T, r, iv)
        ann   = (bid_ret / strike * 100.0) * (365.0 / dte)
        vs_em = (spot - strike) / em * 100.0 if (not math.isnan(em) and em > 0) else float("nan")
        ma_sc = analytics.ma200_score(spot, strike, ma200)
        sc    = score_put(ann, pp, vs_em, ma_sc)

        if sc > best_score:
            best_score = sc
            best = (strike, iv, bid_ret, float(row.get("ask") or 0),
                    _int(row.get("volume")), _int(row.get("openInterest")), delta)
    return best, stats


def find_best_bps(puts: pd.DataFrame, spot: float, T: float, r: float,
                  ma200: float, em: float) -> tuple[Optional[dict], Counter]:
    """Find the bull put spread with the maximum expected value.

    Iterates every viable (short, long) pair and picks the one with the
    highest binary EV = P × max_gain − (1−P) × max_loss. Composite
    `score_put` is still computed for display only.

    Returns (best_dict | None, stats). `best_dict` keys: short_strike,
    short_bid, short_ask, short_iv, short_delta, short_vol, short_oi,
    long_strike, long_bid, long_ask, long_iv, long_delta, width, credit,
    max_loss (per contract), ret_pct (return on margin), ann_rtn,
    profit_prob, ev, score.
    """
    best = None
    best_ev = float("-inf")
    dte = max(T * 365.0, 1.0)
    stats: Counter = Counter()

    oi_unavailable = (
        "openInterest" in puts.columns
        and len(puts) > 0
        and float(puts["openInterest"].fillna(0).max() or 0) == 0
    )
    if oi_unavailable:
        stats["oi_unavailable"] += 1

    _need = [c for c in ("strike", "bid", "ask", "lastPrice", "volume",
                         "openInterest", "lastTradeDate")
             if c in puts.columns]
    rows = puts[_need].to_dict("records")

    # First pass: evaluate liquidity / IV / delta per strike and keep viable ones.
    candidates: list[dict] = []
    for row in rows:
        stats["examined"] += 1
        price_iv, bid_ret, reason = _effective_price(row, oi_unavailable=oi_unavailable)
        if reason:
            stats[reason] += 1
            continue
        strike = float(row["strike"])
        iv = analytics.compute_iv(spot, strike, T, r, price_iv)
        if math.isnan(iv):
            stats["iv_solve"] += 1
            continue
        delta = analytics.bs_put_delta(spot, strike, T, r, iv)
        if math.isnan(delta):
            stats["delta_out"] += 1
            continue
        candidates.append({
            "strike": strike,
            "bid":    bid_ret,
            "ask":    float(row.get("ask") or 0),
            "iv":     iv,
            "delta":  delta,
            "vol":    _int(row.get("volume")),
            "oi":     _int(row.get("openInterest")),
        })
    candidates.sort(key=lambda c: c["strike"])

    # Second pass: pair each candidate short leg with every long leg below it.
    for i, short in enumerate(candidates):
        if not (DELTA_MIN <= abs(short["delta"]) <= DELTA_MAX):
            stats["delta_out"] += 1
            continue
        if MAX_STRIKE and short["strike"] > MAX_STRIKE:
            stats["strike_max"] += 1
            continue
        for j in range(i - 1, -1, -1):
            long_leg = candidates[j]
            width = short["strike"] - long_leg["strike"]
            if width <= 0:
                continue
            if width > BPS_MAX_WIDTH:
                break  # further longs are only wider; abort
            # Long ask must exist (we buy at ask).
            if long_leg["ask"] <= 0:
                continue
            credit = short["bid"] - long_leg["ask"]
            if credit < BPS_MIN_CREDIT:
                continue
            max_loss = (width - credit) * 100.0
            if max_loss <= 0:
                continue
            ret_pct = credit / (width - credit) * 100.0     # return on margin per cycle
            ann     = ret_pct * (365.0 / dte)
            be      = short["strike"] - credit
            pp      = analytics.calc_profit_prob(spot, be, T, r, short["iv"])
            if math.isnan(pp):
                continue
            vs_em   = (spot - short["strike"]) / em * 100.0 if (not math.isnan(em) and em > 0) else float("nan")
            ma_sc   = analytics.ma200_score(spot, short["strike"], ma200)
            sc      = score_put(ann, pp, vs_em, ma_sc)

            # Binary expected value, per contract — the selection criterion.
            ev      = (pp / 100.0) * credit * 100.0 - (1.0 - pp / 100.0) * max_loss

            stats["considered"] += 1
            if ev > best_ev:
                best_ev = ev
                best = {
                    "short_strike": short["strike"],
                    "short_bid":    short["bid"],
                    "short_ask":    short["ask"],
                    "short_iv":     short["iv"],
                    "short_delta":  short["delta"],
                    "short_vol":    short["vol"],
                    "short_oi":     short["oi"],
                    "long_strike":  long_leg["strike"],
                    "long_bid":     long_leg["bid"],
                    "long_ask":     long_leg["ask"],
                    "long_iv":      long_leg["iv"],
                    "long_delta":   long_leg["delta"],
                    "width":        width,
                    "credit":       credit,
                    "max_loss":     max_loss,
                    "ret_pct":      ret_pct,
                    "ann_rtn":      ann,
                    "profit_prob":  pp,
                    "be":           be,
                    "vs_em":        vs_em,
                    "score":        sc,
                    "ev":           ev,
                }
    return best, stats


def _fmt_reject_stats(stats: Counter) -> str:
    """Format a Counter of rejection reasons as a one-line, sorted breakdown."""
    parts = []
    for k, v in stats.most_common():
        if k in REJECT_LABELS and v:
            parts.append(f"{v} {REJECT_LABELS[k]}")
    return ", ".join(parts) or "(no examined strikes)"


# ── Universe loaders ──────────────────────────────────────────────────────────

# Popular option-liquid ETFs: leveraged broad/sector/single-stock + a few
# high-volume non-leveraged thematics. CSP on leveraged ETFs is rough because
# of volatility decay — caveat emptor.
ETF_UNIVERSE = [
    # Leveraged broad index
    "TQQQ", "TNA", "UPRO", "URTY", "FAS",
    # Leveraged sector
    "SOXL", "TECL", "LABU", "GUSH", "ERX",
    # Leveraged single-stock
    "NVDL", "TSLL", "CONL", "MSTU", "AAPB", "AMZU", "AVL", "GGLL",
    # Leveraged crypto
    "BITX", "BITU", "ETHU",
    # High-liquidity non-leveraged benchmarks/sectors/themes
    "SPY", "QQQ", "IWM",
    "SOXX", "SMH", "XLF", "XLE", "XLK", "XLU", "GDX", "ARKK", "USO",
]


def _fetch_sp500() -> list[str]:
    df = pd.read_csv(SP500_URL)
    return [s.replace(".", "-") for s in df["Symbol"].tolist()]


def _fetch_russell2000() -> list[str]:
    import requests
    resp = requests.get(IWM_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("Ticker")), None)
    if start is None:
        raise ValueError("Could not find Ticker column in IWM holdings CSV")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    tickers = df.loc[df["Asset Class"] == "Equity", "Ticker"].dropna().astype(str)
    return [t.replace(".", "-") for t in tickers if t and t not in ("-", "nan")]


def get_tickers() -> list[str]:
    if isinstance(UNIVERSE, list):
        return UNIVERSE
    if UNIVERSE == "sp500":
        return CACHE.fetch("sp500", _fetch_sp500)
    if UNIVERSE == "russell2000":
        return CACHE.fetch("russell2000", _fetch_russell2000)
    if UNIVERSE == "sp500+russell2000":
        sp  = CACHE.fetch("sp500", _fetch_sp500)
        r2k = CACHE.fetch("russell2000", _fetch_russell2000)
        return list(dict.fromkeys(sp + r2k))
    if UNIVERSE == "screener":
        from screener import get_candidates
        return get_candidates(max_price=MAX_STRIKE or 200, verbose=True)
    if UNIVERSE == "etf":
        return list(ETF_UNIVERSE)
    raise ValueError(f"Unknown universe: {UNIVERSE!r}")


# ── Per-ticker scan ───────────────────────────────────────────────────────────

def scan_ticker(symbol: str) -> tuple[Optional[PutRow], Counter]:
    """Scan one ticker.

    Returns (best_row | None, agg_stats). `agg_stats` aggregates the per-
    expiry rejection counters from find_best_put plus ticker-level flags
    (``no_spot``, ``no_expiry``, ``no_chain``, ``examined_expiries``).
    Caller logs / aggregates them.
    """
    agg: Counter = Counter()
    from_cache = CACHE.get(f"fast_info:{symbol}") is not None
    log.debug(f"[{symbol}] {'(cache) ' if from_cache else ''}fetching...")

    meta = CACHE.fetch(f"fast_info:{symbol}", lambda: PROVIDER.get_spot_and_expirations(symbol))
    spot = float(meta.get("price") or 0)
    if not spot or math.isnan(spot) or spot <= 0:
        log.info(f"[{symbol}] no spot price")
        agg["no_spot"] += 1
        return None, agg

    today = datetime.now(timezone.utc).date()

    expiries: list[tuple[int, str]] = []
    all_offered = list(meta.get("options") or [])
    for s in all_offered:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            dte = (d - today).days
            if DTE_MIN <= dte <= DTE_MAX:
                expiries.append((dte, s))
        except ValueError:
            continue

    if not expiries:
        log.info(
            f"[{symbol}] no expiry in {DTE_MIN}-{DTE_MAX} DTE "
            f"(offered: {len(all_offered)} dates)"
        )
        agg["no_expiry"] += 1
        return None, agg

    hist     = CACHE.fetch(f"history:{symbol}",   lambda: PROVIDER.get_history(symbol)) if COMPUTE_IV_RANK else None
    cal      = CACHE.fetch(f"calendar:{symbol}",  lambda: PROVIDER.get_calendar(symbol))
    analyst  = CACHE.fetch(f"analyst:{symbol}",   lambda: PROVIDER.get_analyst_info(symbol))
    fund     = _fund._EMPTY  # fundamentals fetched post-scan for top N only

    ma200     = analytics.get_ma200(hist)
    ma200_pct = _r((spot - ma200) / ma200 * 100.0) if not math.isnan(ma200) else float("nan")
    hv30      = _r(analytics.compute_hv30(hist), 1)
    rsi       = analytics.compute_rsi(hist)
    a_rating  = analyst.get("rating", "")
    a_target  = analyst.get("target", float("nan"))
    a_upside  = _r((a_target - spot) / spot * 100) if not math.isnan(a_target) else float("nan")

    best_row:   Optional[PutRow] = None
    best_score: float = float("-inf")

    for dte, expiry in expiries:
        try:
            chain = CACHE.fetch(
                f"options:{symbol}:{expiry}",
                lambda e=expiry: PROVIDER.get_option_chain(symbol, e),
            )
            calls, puts = chain["calls"], chain["puts"]
            T  = dte / 365.0
            em = analytics.calc_expected_move(calls, puts, spot)

            call_vol = int(calls["volume"].fillna(0).sum()) if not calls.empty and "volume" in calls.columns else 0
            put_vol  = int(puts["volume"].fillna(0).sum())  if not puts.empty  and "volume" in puts.columns  else 0
            pcr  = _r(put_vol / call_vol, 2) if call_vol > 0 else float("nan")
            skew_m = skewlib.compute_skew(calls, puts, spot, T, RISK_FREE_RATE)

            if STRATEGY == "bps":
                result, stats = find_best_bps(puts, spot, T, RISK_FREE_RATE, ma200, em)
            else:
                result, stats = find_best_put(puts, spot, T, RISK_FREE_RATE, ma200, em)
            agg.update(stats)
            agg["examined_expiries"] += 1
            if result is None:
                log.debug(f"[{symbol}] {expiry}: no qualifying put — {_fmt_reject_stats(stats)}")
                continue

            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
            earn      = check_earnings(cal, today, expiry_dt)
            em_pct    = em / spot * 100.0 if not math.isnan(em) else float("nan")

            if STRATEGY == "bps":
                # `result` is a dict from find_best_bps
                b = result
                strike      = b["short_strike"]
                bid         = b["short_bid"]
                ask         = b["short_ask"]
                vol, oi     = b["short_vol"], b["short_oi"]
                iv, delta   = b["short_iv"], b["short_delta"]
                spread      = _r(ask - bid) if (ask > 0 and bid > 0) else float("nan")
                moneyness   = (strike - spot) / spot * 100.0
                be          = b["be"]
                pct_be      = (spot - be) / spot * 100.0
                ret         = b["ret_pct"]
                ann         = b["ann_rtn"]
                pp          = b["profit_prob"]
                vs_em       = b["vs_em"]
                theta       = -analytics.bs_put_theta(spot, strike, T, RISK_FREE_RATE, iv) \
                              + analytics.bs_put_theta(spot, b["long_strike"], T, RISK_FREE_RATE, b["long_iv"])
                sc          = b["score"]
                # EV already computed by find_best_bps as the selection criterion.
                ev          = b["ev"]
                long_extras = dict(
                    long_strike = round(b["long_strike"], 2),
                    long_bid    = round(b["long_bid"], 2),
                    long_ask    = round(b["long_ask"], 2),
                    long_delta  = _r(b["long_delta"], 3),
                    long_iv     = _r(b["long_iv"] * 100, 1),
                    width       = _r(b["width"], 2),
                    credit      = _r(b["credit"], 2),
                    max_loss    = _r(b["max_loss"], 2),
                )
            else:
                strike, iv, bid, ask, vol, oi, delta = result
                spread      = _r(ask - bid) if (ask > 0 and bid > 0) else float("nan")
                moneyness   = (strike - spot) / spot * 100.0
                be          = strike - bid
                pct_be      = (spot - be) / spot * 100.0
                ret         = bid / strike * 100.0
                ann         = ret * (365.0 / dte)
                pp          = analytics.calc_profit_prob(spot, strike, T, RISK_FREE_RATE, iv)
                theta       = -analytics.bs_put_theta(spot, strike, T, RISK_FREE_RATE, iv)
                vs_em       = (spot - strike) / em * 100.0 if (not math.isnan(em) and em > 0) else float("nan")
                ma_sc       = analytics.ma200_score(spot, strike, ma200)
                sc          = score_put(ann, pp, vs_em, ma_sc, fund.score)
                long_extras = {}
                # Binary EV for CSP — pessimistic since max_loss assumes stock→0.
                max_gain    = bid * 100.0
                max_loss_csp = (strike - bid) * 100.0
                ev          = _ev_binary(pp, max_gain, max_loss_csp)

            # ── EV variants (HV30, mid-price, 50%-managed) ─────────────────
            hv30_frac = hv30 / 100.0 if not math.isnan(hv30) else float("nan")
            T_half    = T / 2.0
            if STRATEGY == "bps":
                be_short  = b["be"]
                short_iv  = b["short_iv"]
                p_hv30    = (analytics.calc_profit_prob(spot, be_short, T, RISK_FREE_RATE, hv30_frac)
                             if not math.isnan(hv30_frac) else float("nan"))
                p_half    = (analytics.calc_profit_prob(spot, be_short, T_half, RISK_FREE_RATE, short_iv)
                             if T_half > 0 else float("nan"))
                # Mid-price credit: bid+ask / 2 for short, same for long
                sm = (b["short_bid"] + b["short_ask"]) / 2.0 if (b["short_bid"] > 0 and b["short_ask"] > 0) else b["short_bid"]
                lm = (b["long_bid"]  + b["long_ask"])  / 2.0 if (b["long_bid"]  > 0 and b["long_ask"]  > 0) else b["long_ask"]
                credit_mid   = max(0.0, sm - lm)
                max_loss_mid = (b["width"] - credit_mid) * 100.0 if credit_mid < b["width"] else b["max_loss"]
                ev_hv30      = _ev_binary(p_hv30, b["credit"] * 100.0, b["max_loss"])
                ev_mid       = _ev_binary(pp,     credit_mid * 100.0,  max_loss_mid)
                ev_managed   = _ev_binary(p_half, b["credit"] * 50.0,  b["max_loss"])
            else:
                p_hv30 = (analytics.calc_profit_prob(spot, strike, T, RISK_FREE_RATE, hv30_frac)
                          if not math.isnan(hv30_frac) else float("nan"))
                p_half = (analytics.calc_profit_prob(spot, strike, T_half, RISK_FREE_RATE, iv)
                          if T_half > 0 else float("nan"))
                mid    = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else bid
                ev_hv30    = _ev_binary(p_hv30, bid * 100.0, (strike - bid) * 100.0)
                ev_mid     = _ev_binary(pp,     mid * 100.0, (strike - mid) * 100.0)
                ev_managed = _ev_binary(p_half, bid * 50.0,  (strike - bid) * 100.0)

            ivr = analytics.get_iv_rank(hist, iv) if COMPUTE_IV_RANK else float("nan")

            row = PutRow(
                symbol=symbol,
                price=round(spot, 2),
                exp_date=expiry,
                dte=dte,
                strike=round(strike, 2),
                moneyness=_r(moneyness),
                exp_move=_r(em),
                exp_move_pct=_r(em_pct),
                vs_em=_r(vs_em, 1),
                bid=round(bid, 2),
                ask=round(ask, 2) if ask > 0 else float("nan"),
                spread=spread,
                be_bid=_r(be),
                pct_be_bid=_r(pct_be),
                volume=vol,
                open_int=oi,
                iv_rank=ivr,
                iv=_r(iv * 100),
                hv30=hv30,
                delta=_r(delta, 3),
                theta=_r(theta, 3),
                ret=_r(ret),
                ann_rtn=_r(ann, 1),
                profit_prob=_r(pp, 1),
                earnings_date=earn or "",
                ma200_pct=ma200_pct,
                analyst_rating=a_rating,
                analyst_num=analyst.get("num_analysts", 0),
                analyst_target=_r(a_target, 2),
                analyst_upside=a_upside,
                company_name=analyst.get("company_name", ""),
                fundamental_score=fund.score,
                rev_growth=fund.rev_growth_pct,
                eps_beat_rate=fund.eps_beat_rate,
                fcf_margin=fund.fcf_margin_pct,
                bull_pct=float("nan"),  # fetched post-scan for top N only
                rsi=rsi,
                pcr=pcr,
                rr_25d_pct=skew_m.rr_25d_pct,
                score=_r(sc, 1),
                ev=_r(ev, 2),
                ev_hv30=_r(ev_hv30, 2),
                ev_mid=_r(ev_mid, 2),
                ev_managed=_r(ev_managed, 2),
                **long_extras,
            )

            # For BPS we rank cross-expiry by EV (the per-pair selection
            # criterion); for CSP the composite score_put is still used.
            rank_key = ev if STRATEGY == "bps" else sc
            if rank_key > best_score:
                best_score = rank_key
                best_row = row

        except Exception as e:
            log.error(f"[{symbol}] {expiry}: {e!r}")
            agg["exception"] += 1
            continue

    oi_skipped_expiries = agg.get("oi_unavailable", 0)
    if best_row is None:
        examined = agg.get("examined", 0)
        n_exp    = agg.get("examined_expiries", 0)
        oi_note  = (f" [Yahoo OI=0 across whole chain on {oi_skipped_expiries}/{n_exp} "
                    f"expiries; OI filter bypassed there]" if oi_skipped_expiries else "")
        log.info(
            f"[{symbol}] no qualifying put — examined {examined} strikes across "
            f"{n_exp} expir{'y' if n_exp == 1 else 'ies'}. Filters: "
            f"{_fmt_reject_stats(agg)}{oi_note}"
        )
    else:
        oi_note = " [OI filter bypassed — Yahoo OI=0]" if oi_skipped_expiries else ""
        log.info(f"[{symbol}] OK strike={best_row.strike} dte={best_row.dte} "
                 f"delta={best_row.delta:.3f} ann={best_row.ann_rtn:.1f}%{oi_note}")
    return best_row, agg


# ── STRATEGY=ideas mode: enumerate many strategies per ticker ────────────────

def scan_ticker_ideas(symbol: str) -> tuple[list, Counter]:
    """Enumerate viable trades across every implemented strategy.

    Returns (list[TradeIdea], stats). Each ticker contributes its top-K
    ideas by annualized ROI on margin; caller aggregates and re-ranks.
    """
    agg: Counter = Counter()
    log.debug(f"[{symbol}] fetching...")

    meta = CACHE.fetch(f"fast_info:{symbol}", lambda: PROVIDER.get_spot_and_expirations(symbol))
    spot = float(meta.get("price") or 0)
    if not spot or math.isnan(spot) or spot <= 0:
        log.info(f"[{symbol}] no spot price")
        agg["no_spot"] += 1
        return [], agg

    today = datetime.now(timezone.utc).date()
    expiries: list[tuple[int, str]] = []
    for s in meta.get("options") or []:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            dte = (d - today).days
            if DTE_MIN <= dte <= DTE_MAX:
                expiries.append((dte, s))
        except ValueError:
            continue
    if not expiries:
        log.info(f"[{symbol}] no expiry in DTE range")
        agg["no_expiry"] += 1
        return [], agg

    hist = CACHE.fetch(f"history:{symbol}", lambda: PROVIDER.get_history(symbol)) if COMPUTE_IV_RANK else None
    hv30_pct = analytics.compute_hv30(hist) if hist is not None else float("nan")
    hv30     = hv30_pct / 100.0 if not math.isnan(hv30_pct) else float("nan")

    all_ideas: list = []
    for dte, expiry in expiries:
        try:
            chain = CACHE.fetch(
                f"options:{symbol}:{expiry}",
                lambda e=expiry: PROVIDER.get_option_chain(symbol, e),
            )
            calls, puts = chain["calls"], chain["puts"]
            T = dte / 365.0
            agg["examined_expiries"] += 1

            viable_puts = _strategies.find_candidate_legs(
                puts, spot, T, RISK_FREE_RATE,
                kind="put", eff_price_fn=_effective_price,
                delta_min=DELTA_MIN, delta_max=DELTA_MAX,
                max_strike=MAX_STRIKE,
            )
            viable_calls = _strategies.find_candidate_legs(
                calls, spot, T, RISK_FREE_RATE,
                kind="call", eff_price_fn=_effective_price,
                delta_min=DELTA_MIN, delta_max=DELTA_MAX,
                max_strike=None,  # no cap on call strikes (could add)
            )
            agg["puts_viable"]  += len(viable_puts)
            agg["calls_viable"] += len(viable_calls)
            if not viable_puts and not viable_calls:
                continue

            ideas = _strategies.scan_all(
                viable_puts, viable_calls,
                symbol=symbol, expiry=expiry, dte=dte, spot=spot,
                hv30=hv30, delta_min=DELTA_MIN, delta_max=DELTA_MAX,
                max_width=BPS_MAX_WIDTH, min_credit=BPS_MIN_CREDIT,
                r=RISK_FREE_RATE,
            )
            all_ideas.extend(ideas)
            agg["ideas_examined"] += len(ideas)
        except Exception as e:
            log.error(f"[{symbol}] {expiry}: {e!r}")
            agg["exception"] += 1
            continue

    # Keep top-K per ticker by roi_ann to bound global memory.
    valid = [i for i in all_ideas if not math.isnan(i.roi_ann) and not math.isinf(i.max_loss)]
    valid.sort(key=lambda i: i.roi_ann, reverse=True)
    top = valid[:20]
    if top:
        log.info(f"[{symbol}] {len(all_ideas)} ideas examined, "
                 f"top {len(top)} kept. Best ROI={top[0].roi_ann:+.1f}%/yr "
                 f"({top[0].strategy} {top[0].leg_label()})")
    else:
        log.info(f"[{symbol}] no ideas produced (examined {len(all_ideas)})")
    return top, agg


def print_top_ideas(ideas: list, n: int = 50) -> None:
    """Print top-N trade ideas as an ASCII table."""
    if not ideas:
        log.warning("No trade ideas to display.")
        return
    head = (
        f"\n{'='*132}\n"
        f"Top {min(n, len(ideas))} trade ideas across universe (ranked by EV(mid) annualized ROI on margin)\n"
        f"{'='*132}\n"
        f"  # │ Sym    │ Strat       │ Exp        DTE │ Legs                              "
        f"│   Cr$ │  Max$ │  P%  │ P(HV)% │  EVmid$ │  EVhv30 │  EV50%$ │   ROI%/y\n"
        f"────┼────────┼─────────────┼────────────────┼───────────────────────────────────"
        f"┼───────┼───────┼──────┼────────┼─────────┼─────────┼─────────┼──────────"
    )
    log.info(head)
    for rank, idea in enumerate(ideas[:n], start=1):
        legs_str = idea.leg_label()
        # Truncate legs string to fit column
        if len(legs_str) > 33:
            legs_str = legs_str[:30] + "..."
        log.info(
            f" {rank:>2} │ {idea.symbol:<6} │ {idea.strategy:<11} │ "
            f"{idea.expiry:<10} {idea.dte:>3} │ {legs_str:<33} │ "
            f"{idea.credit:>5.0f} │ {idea.max_loss:>5.0f} │ "
            f"{idea.p_profit:>4.1f} │ {idea.p_profit_hv30:>6.1f} │ "
            f"{idea.ev_mid:>+7.2f} │ {idea.ev_hv30:>+7.2f} │ {idea.ev_managed:>+7.2f} │ "
            f"{idea.roi_ann:>+8.1f}"
        )
    log.info("=" * 132 + "\n")

    # ── Per-strategy top-3 breakdown ────────────────────────────────────────
    by_strat: dict[str, list] = {}
    for i in ideas:
        by_strat.setdefault(i.strategy, []).append(i)
    log.info(f"\n{'='*132}")
    log.info(f"Per-strategy leaderboard (top 3 by ROI/yr, per strategy)")
    log.info("=" * 132)
    log.info(
        f"Strategy         │ N     │ Best 3 candidates (sym / legs / credit / max_loss / P% / EV(mid) / ROI/yr)"
    )
    log.info("─" * 132)
    for strat in sorted(by_strat, key=lambda s: -max((i.roi_ann for i in by_strat[s] if not math.isnan(i.roi_ann)), default=-1e18)):
        picks = sorted(by_strat[strat], key=lambda i: (math.isnan(i.roi_ann), -i.roi_ann if not math.isnan(i.roi_ann) else 0))[:3]
        n_strat = len(by_strat[strat])
        for j, idea in enumerate(picks):
            head = f"{strat:<16}" if j == 0 else " " * 16
            n_str = f"{n_strat:<5}"    if j == 0 else " " * 5
            legs = idea.leg_label()
            if len(legs) > 30:
                legs = legs[:27] + "..."
            roi = f"{idea.roi_ann:+.1f}%/yr" if not math.isnan(idea.roi_ann) else "N/A (∞ loss)"
            log.info(
                f"{head} │ {n_str} │ {idea.symbol:<6} {legs:<32} cr=${idea.credit:>+5.0f} "
                f"maxL=${idea.max_loss:>5.0f} P={idea.p_profit:>4.1f}% "
                f"EVmid=${idea.ev_mid:>+7.2f}  {roi}"
            )
    log.info("=" * 132 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def _scan_link(folder: str, profile: str) -> str:
    """Link to the shared HTML shell for a given scan + profile."""
    return f"_assets/scan.html?scan={folder}&profile={profile}"


def _build_index_html(results_root: Path) -> None:
    from report import _CSS
    entries: list[tuple[str, dict]] = []
    for d in sorted(results_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_file = d / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        entries.append((d.name, meta))

    head_cells = (
        "<th>Scan Time</th><th>Universe</th>"
        "<th>Candidates</th><th>Profiles</th>"
    )
    body_rows: list[str] = []
    for folder, m in entries:
        counts = m.get("counts", {})
        cnts_str = "  ".join(
            f"{p}={counts.get(p, '-')}"
            for p in PROFILES_TO_RUN
        )
        default_link = _scan_link(folder, DEFAULT_PROFILE)
        profile_links = " · ".join(
            f'<a href="{_scan_link(folder, p)}">{p}</a>'
            for p in PROFILES_TO_RUN
        )
        body_rows.append(
            f'<tr>'
            f'<td><a href="{default_link}">{_he.escape(m.get("timestamp", folder))}</a></td>'
            f'<td>{_he.escape(str(m.get("universe", "?")))}</td>'
            f'<td>{_he.escape(cnts_str)}</td>'
            f'<td>{profile_links}</td>'
            f'</tr>'
        )

    if not body_rows:
        body_rows.append('<tr><td colspan="4" class="empty">No scans yet.</td></tr>')

    page = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>CSP Scanner — Scan History</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        "<h1>Short Put Scanner — Scan History</h1>"
        f'<p class="meta">Default link opens the <b>{DEFAULT_PROFILE}</b>-risk report.</p>'
        '<div class="wrap idx-table"><table>'
        f'<thead><tr>{head_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table></div>'
        "</body></html>"
    )
    (results_root / "index.html").write_text(page, encoding="utf-8")
    log.info(f"Updated index → {results_root / 'index.html'}")


def main():
    from cache import DB_PATH
    stats = CACHE.stats()
    total = sum(stats.values()) if stats else 0
    log.info(f"Cache DB : {DB_PATH}")
    log.info(f"Cache    : {dict(stats)}  ({total} entries total)")
    log.info("Loading universe...")
    all_tickers = get_tickers()
    tickers = (
        random.sample(all_tickers, min(SAMPLE_SIZE, len(all_tickers)))
        if SAMPLE_SIZE else all_tickers
    )

    results_root = Path(RESULTS_DIR)
    scan_ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

    counts: dict[str, int] = {}
    scan_data: dict[str, dict] = {}
    for profile_name in PROFILES_TO_RUN:
        globals().update(_PROFILES[profile_name])
        globals()["RISK_PROFILE"] = profile_name
        log.info("\n" + "#" * 70)
        log.info(f"#  PROFILE: {profile_name.upper()}")
        log.info("#" * 70)
        counts[profile_name] = _run_profile(tickers, profile_name, scan_data)

    # Ideas mode is console-only — skip HTML / data.js / index updates.
    if STRATEGY == "ideas":
        return

    scan_dir = results_root / scan_ts
    scan_dir.mkdir(parents=True, exist_ok=True)
    if scan_data:
        write_scan_bundle(scan_dir, scan_data)

    meta = {
        "timestamp": scan_ts,
        "universe": str(UNIVERSE),
        "counts": counts,
    }
    (scan_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _build_index_html(results_root)


def _run_profile(tickers: list[str], profile: str, scan_data: dict) -> int:
    log.info(
        f"Scanning {len(tickers)} tickers  |  profile={RISK_PROFILE.upper()}  "
        f"|  STRATEGY={STRATEGY}  |  DTE {DTE_MIN}-{DTE_MAX}  |  "
        f"|Δ| {DELTA_MIN}-{DELTA_MAX}  |  strike≤{MAX_STRIKE}  |  "
        f"vol≥{MIN_VOLUME}  spread≤{MAX_SPREAD_PCT:.0%}\n"
    )

    # ── STRATEGY=ideas: multi-strategy trade-idea generator ──────────────────
    if STRATEGY == "ideas":
        all_ideas: list = []
        failed: list[str] = []
        global_stats: Counter = Counter()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(scan_ticker_ideas, sym): sym for sym in tickers}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    ideas, stats = fut.result()
                    global_stats.update(stats)
                    all_ideas.extend(ideas)
                except Exception as e:
                    msg = str(e)
                    if "Rate" in msg or "429" in msg:
                        log.warning(f"[{sym}] rate limited — skipping")
                        failed.append(sym)
                    else:
                        log.error(f"[{sym}] error: {e!r}")
        # Global ranking by annualized ROI on margin
        all_ideas.sort(key=lambda i: i.roi_ann, reverse=True)
        log.info(
            f"\nIdeas summary: {len(all_ideas):,} kept across {len(tickers)} tickers. "
            f"Examined puts={global_stats.get('puts_viable', 0):,} "
            f"calls={global_stats.get('calls_viable', 0):,} "
            f"raw ideas={global_stats.get('ideas_examined', 0):,}. "
            f"{len(failed)} skipped."
        )
        print_top_ideas(all_ideas, n=int(os.getenv("IDEAS_TOP_N", "50")))
        return len(all_ideas)

    rows: list[PutRow] = []
    failed: list[str] = []
    rate_errors = 0
    global_stats: Counter = Counter()
    n_no_result = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_ticker, sym): sym for sym in tickers}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                r, stats = fut.result()
                global_stats.update(stats)
                if r:
                    rows.append(r)
                else:
                    n_no_result += 1
                rate_errors = 0
            except Exception as e:
                msg = str(e)
                if "Rate" in msg or "429" in msg:
                    rate_errors += 1
                    wait = min(5 * rate_errors, 60)
                    if wait >= 60:
                        log.warning(f"[{sym}] rate limited — max wait reached, skipping")
                        failed.append(sym)
                        rate_errors = 0
                    else:
                        log.warning(f"[{sym}] rate limited — waiting {wait}s...")
                        time.sleep(wait)
                else:
                    log.error(f"[{sym}] error: {e!r}")
                    n_no_result += 1

    # Per-profile summary: how many tickers found a put, breakdown of why others didn't.
    log.info(
        f"\nProfile {profile.upper()} summary: "
        f"{len(rows)}/{len(tickers)} tickers produced a put "
        f"({n_no_result} with no result, {len(failed)} skipped)."
    )
    if global_stats:
        oi_bypassed = global_stats.get("oi_unavailable", 0)
        log.info(
            f"  Examined {global_stats.get('examined', 0):,} strikes across "
            f"{global_stats.get('examined_expiries', 0):,} expir(ies). "
            f"Ticker-level: "
            f"no_spot={global_stats.get('no_spot', 0)}, "
            f"no_expiry={global_stats.get('no_expiry', 0)}, "
            f"exception={global_stats.get('exception', 0)}."
        )
        if oi_bypassed:
            log.warning(
                f"  ⚠ OI filter auto-bypassed on {oi_bypassed:,} expiries "
                f"(Yahoo returned openInterest=0 for the entire chain — known data quirk)."
            )
        log.info("  Top rejection reasons (across all expiries):")
        for k, v in sorted(global_stats.items(), key=lambda kv: -kv[1]):
            if k in REJECT_LABELS and v:
                log.info(f"    {v:>8,}  {REJECT_LABELS[k]}")

    if not rows:
        log.warning("No results.")
        return 0

    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.sort_values("ann_rtn", ascending=False).reset_index(drop=True)

    def _fmt_ratings(col_rating: pd.Series, col_num: pd.Series) -> pd.Series:
        abbrev = col_rating.map(RATING_ABBREV).fillna(col_rating)
        has_n  = col_num > 0
        return abbrev.where(~has_n, abbrev + " (" + col_num.astype(str) + ")")

    display_map = {
        "symbol":            "Symbol",
        "score":             "Score",
        "price":             "Price~",
        "analyst_rating":    "Rating",
        "analyst_target":    "Target",
        "analyst_upside":    "Upside%",
        "fundamental_score": "FundScore",
        "rev_growth":        "RevGrow%",
        "eps_beat_rate":     "BeatRate%",
        "fcf_margin":        "FCF%",
        "bull_pct":          "ST Bull%",
        "rsi":               "RSI",
        "pcr":               "PCR",
        "rr_25d_pct":        "Skew%",
        "exp_date":          "Exp Date",
        "dte":               "DTE",
        "strike":            "Strike",
        "moneyness":         "Mness%",
        "exp_move_pct":      "EM%",
        "vs_em":             "vs EM%",
        "bid":               "Bid",
        "ask":               "Ask",
        "spread":            "Spread",
        "volume":            "Vol",
        "be_bid":            "BE(Bid)",
        "pct_be_bid":        "%BE",
        "open_int":          "OI",
        "iv_rank":           "IVR",
        "iv":                "IV%",
        "hv30":              "HV30%",
        "delta":             "Delta",
        "theta":             "θ/day",
        "ret":               "Ret%",
        "ann_rtn":           "AnnRtn%",
        "profit_prob":       "PProb%",
        "ma200_pct":         "MA200%",
    }

    disp = df[list(display_map)].rename(columns=display_map).copy()
    disp["Rating"] = _fmt_ratings(df["analyst_rating"], df["analyst_num"])
    disp.loc[df["earnings_date"].ne(""), "Symbol"] += " [!]"

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)

    print(f"\n=== SHORT PUT SCANNER (Cash-Secured Put) ===")
    print(f"Universe: {UNIVERSE}  |  DTE: {DTE_MIN}-{DTE_MAX}  |  |Δ|: {DELTA_MIN}-{DELTA_MAX}\n")
    print(disp.to_string(index=False))

    warn = df[df["earnings_date"].ne("")]
    if not warn.empty:
        print(f"\n[!] EARNINGS WARNING — earnings fall within expiry window:")
        for _, r in warn.iterrows():
            print(f"    {r['symbol']:<6}  earnings={r['earnings_date']}  exp={r['exp_date']}  "
                  f"strike={r['strike']:>8.2f}  delta={r['delta']:>6.3f}  ann_rtn={r['ann_rtn']:>5.1f}%")

    iv_cols = ["symbol", "price", "analyst_rating", "analyst_target", "analyst_upside",
               "exp_date", "strike", "exp_move_pct", "vs_em", "bid", "ask", "spread",
               "volume", "iv_rank", "iv", "hv30", "ann_rtn", "theta", "profit_prob"]
    iv_hdrs = ["Symbol", "Price~", "Rating", "Target", "Upside%",
               "Exp Date", "Strike", "EM%", "vs EM%", "Bid", "Ask", "Spread",
               "Vol", "IVR", "IV%", "HV30%", "AnnRtn%", "θ/day", "PProb%"]

    def _iv_section(subset: pd.DataFrame, title: str) -> None:
        if subset.empty:
            return
        t = subset[iv_cols].rename(columns=dict(zip(iv_cols, iv_hdrs))).copy()
        t["Rating"] = _fmt_ratings(subset["analyst_rating"], subset["analyst_num"])
        print(f"\n--- {title} ---")
        print(t.to_string(index=False))

    no_earn  = df["earnings_date"].eq("")
    high_ivr = df["iv_rank"] >= 50
    iv_gt_hv = df["iv"] > df["hv30"]

    _iv_section(df[ high_ivr & ~iv_gt_hv & no_earn].head(20),
                "HIGH IVR (>=50) only  — options elevated vs own history")
    _iv_section(df[~high_ivr &  iv_gt_hv & no_earn].head(20),
                "IV > HV30 only  — options priced above realized vol")
    _iv_section(df[ high_ivr &  iv_gt_hv & no_earn].head(20),
                "HIGH IVR + IV > HV30  — best premium candidates")

    top_rows = sorted(rows, key=lambda r: r.score, reverse=True)[:AI_TOP_N]

    fund_raw: dict[str, dict] = {}
    processed: list = []
    print(f"\nFetching sentiment + fundamentals for top {len(top_rows)} "
          f"(batch size {AI_BATCH_SIZE})...")
    for batch_start in range(0, len(top_rows), AI_BATCH_SIZE):
        batch = top_rows[batch_start : batch_start + AI_BATCH_SIZE]
        if batch_start > 0 and not NON_INTERACTIVE:
            remaining = len(top_rows) - batch_start
            ans = input(f"  Processed {batch_start}. Fetch next "
                        f"{min(AI_BATCH_SIZE, remaining)}? [y/N]: ").strip().lower()
            if ans != "y":
                print(f"  Stopping at {batch_start} candidates.")
                break

        for r in batch:
            sent = CACHE.fetch(f"sentiment:{r.symbol}",
                               lambda s=r.symbol: _sentiment.fetch_sentiment(s))
            r.bull_pct = sent.get("bull_pct", float("nan"))
            df.loc[df["symbol"] == r.symbol, "bull_pct"] = r.bull_pct

            if ENABLE_FUNDAMENTALS and _fund.CLIENT:
                fm = _fund.fetch_and_compute(r.symbol, CACHE)
                r.fundamental_score = fm.score
                r.rev_growth        = fm.rev_growth_pct
                r.eps_beat_rate     = fm.eps_beat_rate
                r.fcf_margin        = fm.fcf_margin_pct
                mask = df["symbol"] == r.symbol
                df.loc[mask, "fundamental_score"] = fm.score
                df.loc[mask, "rev_growth"]        = fm.rev_growth_pct
                df.loc[mask, "eps_beat_rate"]     = fm.eps_beat_rate
                df.loc[mask, "fcf_margin"]        = fm.fcf_margin_pct
                sym = r.symbol
                news = CACHE.fetch(f"fd_news:{sym}", lambda s=sym: _fund.CLIENT.fetch_news(s))
                fund_raw[sym] = {
                    "income":   CACHE.get(f"fd_income:{sym}")   or [],
                    "cashflow": CACHE.get(f"fd_cashflow:{sym}") or [],
                    "balance":  CACHE.get(f"fd_balance:{sym}")  or [],
                    "earnings": CACHE.get(f"fd_earnings:{sym}") or [],
                    "news":     news,
                }
            processed.append(r)

    top_rows = processed

    ai_text = None
    if ENABLE_AI_ANALYSIS and top_rows:
        if len(top_rows) > AI_BATCH_SIZE and not NON_INTERACTIVE:
            ans = input(f"\nSend {len(top_rows)} candidates to {AI_MODEL}? [y/N]: ").strip().lower()
            if ans != "y":
                print("  Skipping AI analysis.")
            else:
                ai_text = ai_analysis(top_rows, fund_raw, model=AI_MODEL)
        else:
            ai_text = ai_analysis(top_rows, fund_raw, model=AI_MODEL)
    if ai_text:
        print(f"\n{'='*60}")
        print(f"AI ANALYSIS (top {len(top_rows)} by score)")
        print('='*60)
        print(ai_text)

    config_str = (f"Universe: {UNIVERSE}  |  Profile: {RISK_PROFILE}  |  "
                  f"DTE: {DTE_MIN}-{DTE_MAX}  |  |Δ|: {DELTA_MIN}-{DELTA_MAX}  |  "
                  f"strike≤{MAX_STRIKE}  |  vol≥{MIN_VOLUME}  |  spread≤{MAX_SPREAD_PCT:.0%}")
    scan_data[profile] = build_profile_block(
        df, config_str, failed,
        ai_text=ai_text, ai_top_n=AI_TOP_N,
        profile=profile, index_href="../index.html", strategy=STRATEGY,
    )
    return len(df)

    if failed:
        print(f"\n[!] {len(failed)} tickers skipped due to rate limiting:")
        print("    " + ", ".join(failed))
        print(f"    Re-run with UNIVERSE = {failed!r} to retry them.")

    return len(df)


if __name__ == "__main__":
    main()
