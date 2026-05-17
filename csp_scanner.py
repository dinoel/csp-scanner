"""
Short Put (Cash-Secured Put) scanner for US equity options via yfinance (delayed).

For each ticker in the universe, finds the put option within DTE_MIN-DTE_MAX
closest to TARGET_DELTA and reports premium, break-even, return, and
probability of profit. Warns when earnings fall within the expiry window.

Usage:
    python csp_scanner.py
todo: filter based on volume/spread , use mid price instead of bid
"""

from __future__ import annotations

import io
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from cache import CACHE

# ── CONFIG ────────────────────────────────────────────────────────────────────
#UNIVERSE: str | list[str] = ["NVDA", "INTC", "AMD", "PLTR", "MCD", "ASTS", "TEAM", "NBIS", "IREN", "EMR", "ORCL", "DELL", "SMCI", "HAL", "GLW", "FCX", "HUT", "ARM"]
#UNIVERSE: str | list[str] = ["CRWV"]
UNIVERSE: str | list[str] = "screener"
SAMPLE_SIZE: int | None = None        # tickers to sample; None = full universe
RISK_PROFILE = "medium"              # "low" | "medium" | "high"
RISK_FREE_RATE = 0.05
MIN_BID = 0.05
COMPUTE_IV_RANK = True
MAX_WORKERS = 6      # parallel ticker scans; keep ≤8 to avoid Yahoo rate limits
CSV_OUT  = "csp_scan.csv"
HTML_OUT = "csp_scan.html"
ENABLE_AI_ANALYSIS = True    # set False or unset ANTHROPIC_API_KEY to skip
AI_TOP_N = 10                 # how many top candidates to send to Claude
AI_MODEL = "claude-opus-4-7"

# ── Risk profiles ─────────────────────────────────────────────────────────────
#
#  low    — conservative: far OTM, longer DTE, strict liquidity, safety-first
#  medium — balanced (default)
#  high   — aggressive: closer ATM, short DTE, return-first
#
#  Each profile sets: DTE_MIN/MAX · DELTA_MIN/MAX · MAX_STRIKE
#                     MIN_OPEN_INTEREST · MIN_VOLUME · MAX_SPREAD_PCT
#                     SCORE_W_PROB · SCORE_W_RETURN · SCORE_W_SAFETY · SCORE_W_MA200
#
_PROFILES: dict[str, dict] = {
    "low": dict(
        DTE_MIN=21,  DTE_MAX=51,
        DELTA_MIN=0.05, DELTA_MAX=0.20,   # very OTM — PProb 85-95%
        MAX_STRIKE=150,
        MIN_OPEN_INTEREST=100, MIN_VOLUME=25, MAX_SPREAD_PCT=0.30,
        SCORE_W_PROB=0.55, SCORE_W_RETURN=0.15, SCORE_W_SAFETY=0.20, SCORE_W_MA200=0.10,
    ),
    "medium": dict(
        DTE_MIN=7,   DTE_MAX=51,
        DELTA_MIN=0.05, DELTA_MAX=0.50,
        MAX_STRIKE=200,
        MIN_OPEN_INTEREST=50,  MIN_VOLUME=200,  MAX_SPREAD_PCT=0.50,
        SCORE_W_PROB=0.45, SCORE_W_RETURN=0.25, SCORE_W_SAFETY=0.20, SCORE_W_MA200=0.10,
    ),
    "high": dict(
        DTE_MIN=7,   DTE_MAX=31,           # shorter DTE → higher annualized return
        DELTA_MIN=0.15, DELTA_MAX=0.50,    # closer ATM — more premium, more risk
        MAX_STRIKE=200,
        MIN_OPEN_INTEREST=50,  MIN_VOLUME=200,  MAX_SPREAD_PCT=0.30,
        SCORE_W_PROB=0.30, SCORE_W_RETURN=0.45, SCORE_W_SAFETY=0.15, SCORE_W_MA200=0.10,
    ),
}

if RISK_PROFILE not in _PROFILES:
    raise ValueError(f"Unknown RISK_PROFILE {RISK_PROFILE!r} — choose: {list(_PROFILES)}")
globals().update(_PROFILES[RISK_PROFILE])
# ─────────────────────────────────────────────────────────────────────────────

SP500_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
    "/refs/heads/main/data/constituents.csv"
)


@dataclass
class PutRow:
    symbol: str
    price: float
    exp_date: str
    dte: int
    strike: float
    moneyness: float      # (strike-price)/price*100; negative = OTM
    exp_move: float       # ATM straddle price (expected move in $)
    exp_move_pct: float   # EM as % of spot
    vs_em: float          # (spot-strike)/EM*100; 100=at boundary, >100=outside
    bid: float
    ask: float
    spread: float         # ask - bid (NaN when only lastPrice available)
    be_bid: float         # strike - bid  (break-even price)
    pct_be_bid: float     # (price - be_bid)/price*100  (downside cushion %)
    volume: int
    open_int: int
    iv_rank: float        # 0-100 (NaN if unavailable)
    iv: float             # put IV at strike
    hv30: float           # 30-day realized vol, annualized %
    delta: float          # negative (put delta)
    theta: float          # daily $ gain for seller (positive)
    ret: float            # bid/strike*100  (return on capital %)
    ann_rtn: float        # annualized return %
    profit_prob: float    # P(expires OTM) %
    earnings_date: str    # earnings date 'YYYY-MM-DD' if in window, else ''
    ma200_pct: float      # (spot - MA200) / MA200 * 100; positive = above MA200
    analyst_rating: str   # consensus: "strong_buy"|"buy"|"hold"|"underperform"|"sell"|""
    analyst_num: int      # number of analysts covering the stock
    analyst_target: float # analyst mean price target
    analyst_upside: float # (target - price) / price * 100
    score: float          # composite score (higher = better)


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_put_delta(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1, _ = _d1d2(S, K, T, r, sigma)
    return _ncdf(d1) - 1.0


def bs_put_price(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def compute_iv(S, K, T, r, price) -> float:
    """Put IV via bisection from a market price. Returns NaN on failure."""
    intrinsic = max(0.0, (K - S) * math.exp(-r * T))
    if price <= intrinsic or price <= 0:
        return float("nan")
    lo, hi = 1e-4, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_put_price(S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    iv = (lo + hi) / 2
    return iv if 0.01 < iv < 5.0 else float("nan")


def calc_profit_prob(S, K, T, r, iv) -> float:
    """Risk-neutral P(S_T > K) = N(d2); probability the put expires worthless."""
    if T <= 0 or iv <= 0:
        return float("nan")
    _, d2 = _d1d2(S, K, T, r, iv)
    return _ncdf(d2) * 100.0


def _ma200_score(spot: float, strike: float, ma200: float) -> float:
    """0-100 score for the MA200 component.

    Rewards uptrend (spot > MA200) and extra safety buffer
    when the strike sits below MA200 (the 200-day acts as support)."""
    if math.isnan(ma200) or ma200 <= 0:
        return 50.0                          # neutral when data unavailable
    pct = (spot - ma200) / ma200 * 100.0    # % above/below MA200
    if   pct >= 10: trend = 90
    elif pct >=  5: trend = 75
    elif pct >=  0: trend = 60
    elif pct >= -5: trend = 35
    elif pct >= -10: trend = 20
    else:           trend = 10
    # Bonus: strike below MA200 while stock is above it → MA200 cushion beneath us
    bonus = 15 if (spot > ma200 and strike < ma200) else 0
    return min(float(trend + bonus), 100.0)


def score_put(ann_rtn: float, profit_prob: float, vs_em: float,
              ma200_sc: float = 50.0) -> float:
    """Composite score for a (strike, expiry) candidate. Higher = better."""
    if math.isnan(ann_rtn) or math.isnan(profit_prob):
        return float("-inf")
    exp_rtn = min(ann_rtn * profit_prob / 100.0, 200.0)
    em_buf  = min(max(0.0 if math.isnan(vs_em) else vs_em, 0.0), 150.0)
    return (SCORE_W_RETURN * exp_rtn +
            SCORE_W_PROB   * profit_prob +
            SCORE_W_SAFETY * em_buf +
            SCORE_W_MA200  * ma200_sc)


def bs_put_theta(S, K, T, r, sigma) -> float:
    """Daily theta for a long put (negative). Seller earns -theta per share per day."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1, d2 = _d1d2(S, K, T, r, sigma)
    pdf_d1 = math.exp(-d1 ** 2 / 2) / math.sqrt(2 * math.pi)
    annual = (
        -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
        + r * K * math.exp(-r * T) * _ncdf(-d2)
    )
    return annual / 365


def _int(v) -> int:
    """int() safe for NaN/None (e.g. volume column after JSON round-trip)."""
    try:
        f = float(v)
        return 0 if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return 0


def _mid_or_last(row) -> float:
    """Best available price (mid > bid > last), no OI filter — used for EM straddle."""
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    last = float(row.get("lastPrice") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return bid if bid > 0 else (last if last > 0 else float("nan"))


def calc_expected_move(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    """ATM straddle price = ATM call mid + ATM put mid. Market's 1-SD expected move."""
    strikes = set(calls["strike"].dropna().values) & set(puts["strike"].dropna().values)
    if not strikes:
        return float("nan")
    atm = min(strikes, key=lambda k: abs(k - spot))
    call_rows = calls[calls["strike"] == atm]
    put_rows  = puts[puts["strike"] == atm]
    if call_rows.empty or put_rows.empty:
        return float("nan")
    c = _mid_or_last(call_rows.iloc[0])
    p = _mid_or_last(put_rows.iloc[0])
    return float("nan") if (math.isnan(c) or math.isnan(p)) else c + p


# ── Data helpers ──────────────────────────────────────────────────────────────

def pick_expiry(expirations: list[str], today, dte_min: int, dte_max: int):
    candidates = []
    for s in expirations:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            dte = (d - today).days
            if dte_min <= dte <= dte_max:
                candidates.append((dte, s))
        except ValueError:
            continue
    if not candidates:
        return None, None
    dte, s = min(candidates)
    return s, dte


def _effective_price(row) -> tuple[float, float]:
    """Return (price_for_iv, bid_for_return) or (nan, nan) if liquidity filters fail.

    Filters applied:
      - openInterest >= MIN_OPEN_INTEREST
      - volume >= MIN_VOLUME  (on any price source)
      - (ask-bid)/bid <= MAX_SPREAD_PCT  (live quotes only)
    Price priority: mid (live bid+ask) > bid > lastPrice (fallback)."""
    oi = _int(row.get("openInterest"))
    if oi < MIN_OPEN_INTEREST:
        return float("nan"), float("nan")

    vol = _int(row.get("volume"))
    if vol < MIN_VOLUME:
        return float("nan"), float("nan")

    bid  = float(row.get("bid")       or 0)
    ask  = float(row.get("ask")       or 0)
    last = float(row.get("lastPrice") or 0)

    if bid > 0 and ask > 0:
        if (ask - bid) / bid > MAX_SPREAD_PCT:
            return float("nan"), float("nan")   # spread too wide — price unreliable
        return (bid + ask) / 2, bid

    if bid > MIN_BID:
        return bid, bid

    if last > MIN_BID:
        # After-hours / weekend fallback: no live spread check possible.
        # Apply a stricter volume threshold to reduce stale-price risk.
        if vol < max(MIN_VOLUME * 5, 50):
            return float("nan"), float("nan")
        return last, last

    return float("nan"), float("nan")


def get_ma200(hist: pd.DataFrame) -> float:
    """200-day simple moving average of closing price."""
    try:
        if hist is None or len(hist) < 200:
            return float("nan")
        return float(hist["Close"].squeeze().rolling(200).mean().iloc[-1])
    except Exception:
        return float("nan")


def find_best_put(puts: pd.DataFrame, spot: float, T: float, r: float,
                  ma200: float, em: float) -> Optional[tuple]:
    """Score every eligible put strike and return the best one.

    Eligible = OI/bid filters pass AND DELTA_MIN ≤ |delta| ≤ DELTA_MAX.
    Returns (strike, iv, bid, ask, volume, open_int, delta) or None."""
    best = None
    best_score = float("-inf")
    dte = max(T * 365.0, 1.0)

    # to_dict("records") → plain Python dicts; ~10x faster than iterrows()
    # which creates a full pandas Series per row with heavy type-checking overhead
    _need = [c for c in ("strike","bid","ask","lastPrice","volume","openInterest")
             if c in puts.columns]
    for row in puts[_need].to_dict("records"):
        price_iv, bid_ret = _effective_price(row)
        if math.isnan(price_iv) or math.isnan(bid_ret):
            continue
        strike = float(row["strike"])
        iv = compute_iv(spot, strike, T, r, price_iv)
        if math.isnan(iv):
            continue
        delta = bs_put_delta(spot, strike, T, r, iv)
        if math.isnan(delta) or not (DELTA_MIN <= abs(delta) <= DELTA_MAX):
            continue
        if MAX_STRIKE and strike > MAX_STRIKE:
            continue

        pp     = calc_profit_prob(spot, strike, T, r, iv)
        ann    = (bid_ret / strike * 100.0) * (365.0 / dte)
        vs_em  = (spot - strike) / em * 100.0 if (not math.isnan(em) and em > 0) else float("nan")
        ma_sc  = _ma200_score(spot, strike, ma200)
        sc     = score_put(ann, pp, vs_em, ma_sc)

        if sc > best_score:
            best_score = sc
            best = (strike, iv, bid_ret, float(row.get("ask") or 0),
                    _int(row.get("volume")), _int(row.get("openInterest")), delta)
    return best


def get_iv_rank(hist: pd.DataFrame, current_iv: float) -> float:
    """IV rank: where current ATM IV sits in 52-week rolling-HV range (0–100).
    Uses 21-day rolling historical volatility as a proxy for IV history."""
    if not COMPUTE_IV_RANK:
        return float("nan")
    try:
        if hist is None or len(hist) < 30:
            return float("nan")
        close = hist["Close"].squeeze()
        log_ret = np.log(close / close.shift(1)).dropna()
        hv = log_ret.rolling(21).std() * math.sqrt(252)
        hv = hv.dropna()
        if hv.empty:
            return float("nan")
        lo, hi = float(hv.min()), float(hv.max())
        if hi <= lo:
            return float("nan")
        rank = (current_iv - lo) / (hi - lo) * 100.0
        return round(float(max(0.0, min(100.0, rank))), 1)
    except Exception:
        return float("nan")


def compute_hv30(hist: pd.DataFrame) -> float:
    """30-day annualized historical volatility (realized vol) in %."""
    try:
        if hist is None or len(hist) < 31:
            return float("nan")
        close = hist["Close"].squeeze()
        log_ret = np.log(close / close.shift(1)).dropna()
        return float(log_ret.iloc[-30:].std() * math.sqrt(252) * 100)
    except Exception:
        return float("nan")


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


# ── yfinance fetchers (called by cache on miss) ───────────────────────────────

def _fetch_spot_options(symbol: str) -> dict:
    tk = yf.Ticker(symbol)
    # fast_info.last_price internally downloads 1yr of history in yfinance 1.3 —
    # use a minimal 5-day history call instead (one small HTTP request vs one large one)
    try:
        price = float(tk.history(period="5d", auto_adjust=True)["Close"].iloc[-1])
    except Exception:
        price = 0.0
    return {"price": price, "options": list(tk.options or [])}

def _fetch_chain(symbol: str, expiry: str) -> dict:
    chain = yf.Ticker(symbol).option_chain(expiry)
    return {"calls": chain.calls, "puts": chain.puts}

def _fetch_history(symbol: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)

def _fetch_calendar(symbol: str) -> dict:
    cal = yf.Ticker(symbol).calendar
    if isinstance(cal, pd.DataFrame):
        return cal.to_dict()
    return cal or {}

def _fetch_sp500() -> list[str]:
    df = pd.read_csv(SP500_URL)
    return [s.replace(".", "-") for s in df["Symbol"].tolist()]


IWM_URL = (
    "https://www.ishares.com/us/products/239710/x"
    "/1467271812596.ajax?fileType=csv&dataType=fund"
)

def _fetch_analyst(symbol: str) -> dict:
    info = yf.Ticker(symbol).info
    target_raw = info.get("targetMeanPrice")
    return {
        "rating":       info.get("recommendationKey") or "",
        "target":       float(target_raw) if target_raw else float("nan"),
        "num_analysts": int(info.get("numberOfAnalystOpinions") or 0),
    }


def _fetch_russell2000() -> list[str]:
    import requests
    resp = requests.get(IWM_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    # iShares CSV has several metadata rows before the actual header
    start = next((i for i, l in enumerate(lines) if l.startswith("Ticker")), None)
    if start is None:
        raise ValueError("Could not find Ticker column in IWM holdings CSV")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    tickers = df.loc[df["Asset Class"] == "Equity", "Ticker"].dropna().astype(str)
    return [t.replace(".", "-") for t in tickers if t and t not in ("-", "nan")]


# ── Per-ticker scan ───────────────────────────────────────────────────────────

def _r(x, n: int = 2) -> float:
    """round() safe for NaN."""
    try:
        f = float(x)
        return round(f, n) if not math.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def scan_ticker(symbol: str) -> Optional[PutRow]:
    from_cache = CACHE.get(f"fast_info:{symbol}") is not None
    print(f"[{symbol}] {'(cache) ' if from_cache else ''}fetching...")

    meta = CACHE.fetch(f"fast_info:{symbol}", lambda: _fetch_spot_options(symbol))
    spot = float(meta.get("price") or 0)
    if not spot or math.isnan(spot) or spot <= 0:
        return None

    today = datetime.now(timezone.utc).date()

    # All valid expirations in the DTE window
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
        print(f"[{symbol}] no expiry in {DTE_MIN}-{DTE_MAX} DTE")
        return None

    # Fetch once — shared across all expiries
    hist      = CACHE.fetch(f"history:{symbol}", lambda: _fetch_history(symbol)) if COMPUTE_IV_RANK else None
    cal       = CACHE.fetch(f"calendar:{symbol}", lambda: _fetch_calendar(symbol))
    analyst   = CACHE.fetch(f"analyst:{symbol}", lambda: _fetch_analyst(symbol))
    ma200     = get_ma200(hist)
    ma200_pct = _r((spot - ma200) / ma200 * 100.0) if not math.isnan(ma200) else float("nan")
    hv30      = _r(compute_hv30(hist), 1)
    a_rating  = analyst.get("rating", "")
    a_target  = analyst.get("target", float("nan"))
    a_upside  = _r((a_target - spot) / spot * 100) if not math.isnan(a_target) else float("nan")

    best_row:   Optional[PutRow] = None
    best_score: float = float("-inf")

    for dte, expiry in expiries:
        try:
            chain = CACHE.fetch(
                f"options:{symbol}:{expiry}",
                lambda e=expiry: _fetch_chain(symbol, e),
            )
            calls, puts = chain["calls"], chain["puts"]
            T  = dte / 365.0
            em = calc_expected_move(calls, puts, spot)

            result = find_best_put(puts, spot, T, RISK_FREE_RATE, ma200, em)
            if result is None:
                continue

            strike, iv, bid, ask, vol, oi, delta = result
            spread = _r(ask - bid) if (ask > 0 and bid > 0) else float("nan")

            moneyness = (strike - spot) / spot * 100.0
            be        = strike - bid
            pct_be    = (spot - be) / spot * 100.0
            ret       = bid / strike * 100.0
            ann       = ret * (365.0 / dte)
            pp        = calc_profit_prob(spot, strike, T, RISK_FREE_RATE, iv)
            theta     = -bs_put_theta(spot, strike, T, RISK_FREE_RATE, iv)
            em_pct    = em / spot * 100.0 if not math.isnan(em) else float("nan")
            vs_em     = (spot - strike) / em * 100.0 if (not math.isnan(em) and em > 0) else float("nan")
            ivr       = get_iv_rank(hist, iv)
            ma_sc     = _ma200_score(spot, strike, ma200)
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
            earn      = check_earnings(cal, today, expiry_dt)
            sc        = score_put(ann, pp, vs_em, ma_sc)

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
                score=_r(sc, 1),
            )

            if sc > best_score:
                best_score = sc
                best_row = row

        except Exception:
            continue

    if best_row is None:
        print(f"[{symbol}] no qualifying put")
    return best_row


# ── Universe loader ───────────────────────────────────────────────────────────

def get_tickers() -> list[str]:
    if isinstance(UNIVERSE, list):
        return UNIVERSE
    if UNIVERSE == "sp500":
        return CACHE.fetch("sp500", _fetch_sp500)
    if UNIVERSE == "russell2000":
        return CACHE.fetch("russell2000", _fetch_russell2000)
    if UNIVERSE == "sp500+russell2000":
        sp = CACHE.fetch("sp500", _fetch_sp500)
        r2k = CACHE.fetch("russell2000", _fetch_russell2000)
        return list(dict.fromkeys(sp + r2k))  # merge, deduplicate, preserve order
    if UNIVERSE == "screener":
        from screener import get_candidates
        return get_candidates(max_price=MAX_STRIKE or 200, verbose=True)
    raise ValueError(f"Unknown universe: {UNIVERSE!r}")


# ── HTML output ───────────────────────────────────────────────────────────────

# Analyst rating maps — shared by terminal and HTML renderers
_RATING_ABBREV = {
    "strong_buy": "STR_BUY", "buy": "BUY", "hold": "HOLD",
    "underperform": "UNDP",   "sell": "SELL",
}
_RATING_CLASS = {
    "strong_buy": "r-sb", "buy": "r-b", "hold": "r-h",
    "underperform": "r-s", "sell": "r-s",
}

_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font:12px/1.5 'Consolas','Courier New',monospace;padding:20px}
h1{color:#58a6ff;font-size:16px;margin-bottom:4px}
h2{color:#3fb950;font-size:12px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid #21262d;text-transform:uppercase;letter-spacing:.05em}
.meta{color:#8b949e;font-size:11px;margin-bottom:20px}
.wrap{overflow-x:auto;margin-bottom:4px}
table{border-collapse:collapse;white-space:nowrap}
thead th{background:#161b22;color:#58a6ff;padding:5px 10px;cursor:pointer;user-select:none;border-bottom:2px solid #30363d;text-align:right;position:sticky;top:0;z-index:1}
thead th:first-child{text-align:left}
thead th:hover{background:#1f2937}
thead th.asc::after{content:' ▲';font-size:9px;opacity:.8}
thead th.desc::after{content:' ▼';font-size:9px;opacity:.8}
tbody td{padding:3px 10px;text-align:right;border-bottom:1px solid #161b22;color:#e6edf3}
tbody td:first-child{text-align:left}
tbody tr:hover td{filter:brightness(1.5)}
.earn{color:#f0883e!important;font-weight:700}
.r-sb{color:#56d364;font-weight:700}.r-b{color:#3fb950}
.r-h{color:#d29922}.r-s{color:#f85149}
.empty{color:#6e7681;font-style:italic;font-size:11px;padding:4px 0}
.failed{color:#f85149;font-size:11px;margin-top:24px}
.ai-box{background:#111827;border:1px solid #30363d;border-left:3px solid #58a6ff;
        border-radius:4px;padding:14px 18px;margin:8px 0 4px;max-width:900px;
        font-size:12px;line-height:1.7;color:#c9d1d9}
.ai-box h3{color:#58a6ff;font-size:12px;margin:10px 0 4px}
.ai-box strong{color:#e6edf3}
.ai-box p{margin:0 0 6px}
"""

_JS = """\
document.querySelectorAll('table[id]').forEach(t=>{
  const hs=[...t.querySelectorAll('thead th')];let s={c:-1,a:1};
  hs.forEach((h,i)=>{
    h.addEventListener('click',()=>{
      const a=s.c===i?-s.a:1;s={c:i,a};
      hs.forEach(x=>x.classList.remove('asc','desc'));
      h.classList.add(a>0?'asc':'desc');
      const tb=t.querySelector('tbody');
      [...tb.rows].sort((x,y)=>{
        const av=x.cells[i].dataset.v,bv=y.cells[i].dataset.v;
        const an=parseFloat(av),bn=parseFloat(bv);
        const d=(!isNaN(an)&&!isNaN(bn))?an-bn:String(av).localeCompare(String(bv));
        return a*d;
      }).forEach(r=>tb.append(r));
    });
  });
});
"""


def _score_bg(score: float, lo: float, hi: float) -> str:
    """Row background: #111827 (dark) → #15803d (green) by normalized score."""
    try:
        t = 0.0 if hi <= lo else max(0.0, min(1.0, (float(score) - lo) / (hi - lo)))
    except (TypeError, ValueError):
        t = 0.0
    return f"rgb({int(17+t*4)},{int(24+t*104)},{int(39+t*22)})"


def _html_table(subset: pd.DataFrame, fields: list, headers: list,
                table_id: str, s_lo: float, s_hi: float) -> str:
    """Return a sortable HTML <table> string for the given DataFrame subset."""
    import html as _he
    if subset.empty:
        return '<p class="empty">No results in this category.</p>'

    head = ("<thead><tr>"
            + "".join(f"<th>{_he.escape(h)}</th>" for h in headers)
            + "</tr></thead>")
    rows = []
    for rec in subset.to_dict("records"):
        is_earn = bool(rec.get("earnings_date", ""))
        bg      = _score_bg(rec.get("score", float("nan")), s_lo, s_hi)
        cells   = []
        for f in fields:
            v = rec.get(f)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append('<td data-v="-9999">—</td>')
            elif f == "symbol":
                s = _he.escape(str(v))
                if is_earn:
                    cells.append(f'<td data-v="{s}" class="earn">{s} [!]</td>')
                else:
                    cells.append(f'<td data-v="{s}">{s}</td>')
            elif f == "analyst_rating":
                raw  = str(v)
                abbr = _RATING_ABBREV.get(raw, raw)
                n    = int(rec.get("analyst_num") or 0)
                disp = _he.escape(f"{abbr} ({n})" if abbr and n > 0 else abbr or "—")
                cls  = _RATING_CLASS.get(raw, "")
                ca   = f' class="{cls}"' if cls else ""
                cells.append(f'<td data-v="{_he.escape(raw)}"{ca}>{disp}</td>')
            elif isinstance(v, int):
                cells.append(f'<td data-v="{v}">{v:,}</td>')
            elif isinstance(v, float):
                cells.append(f'<td data-v="{v}">{v}</td>')
            else:
                cells.append(f'<td data-v="-9999">{_he.escape(str(v))}</td>')
        rows.append(f'<tr style="background:{bg}">{"".join(cells)}</tr>')

    body = "<tbody>" + "".join(rows) + "</tbody>"
    return f'<div class="wrap"><table id="{table_id}">{head}{body}</table></div>'


def ai_analysis(top_rows: list) -> str | None:
    """Send top N put candidates to Claude for qualitative analysis.

    Requires: pip install anthropic  +  ANTHROPIC_API_KEY env var.
    Returns the analysis text, or None if unavailable."""
    if not ENABLE_AI_ANALYSIS:
        return None
    try:
        import anthropic
    except ImportError:
        print("[AI] 'anthropic' not installed — run: pip install anthropic")
        return None
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[AI] ANTHROPIC_API_KEY not set — skipping AI analysis")
        return None

    lines = [
        "You are a quantitative options analyst. Analyze these cash-secured put (short put) "
        "candidates for a retail trader. Data is from a systematic scanner using Yahoo Finance "
        "(15–20 min delayed).\n\n"
        "Candidates (sorted by composite score, best first):\n\n"
    ]
    for i, r in enumerate(top_rows, 1):
        earn = f"⚠ Earnings {r.earnings_date} within window" if r.earnings_date else "No earnings in window"
        rating_str = f"{r.analyst_rating} ({r.analyst_num} analysts)" if r.analyst_rating else "n/a"
        lines.append(
            f"{i}. {r.symbol}  —  Put ${r.strike}  exp {r.exp_date}  DTE {r.dte}\n"
            f"   Price ${r.price}  |  Bid ${r.bid}  |  Ann return {r.ann_rtn}%  |  Profit prob {r.profit_prob}%\n"
            f"   IV {r.iv}%  |  HV30 {r.hv30}%  |  IVR {r.iv_rank}  |  Delta {r.delta}\n"
            f"   Moneyness {r.moneyness:+.1f}%  |  vs EM {r.vs_em:.0f}%  |  MA200 {r.ma200_pct:+.1f}%\n"
            f"   Analyst: {rating_str}  |  Target ${r.analyst_target}  |  Upside {r.analyst_upside:+.1f}%\n"
            f"   Score {r.score}  |  {earn}\n\n"
        )
    lines.append(
        "For each position provide:\n"
        "- 2-3 sentences: key attractiveness factors + main risk(s)\n"
        "- Risk: Low / Medium / High\n"
        "- Verdict: Recommended / Neutral / Avoid\n\n"
        "End with a 2-sentence overall market observation implied by these results.\n"
        "Be concise and actionable. No generic disclaimers."
    )

    print(f"[AI] Requesting analysis of top {len(top_rows)} candidates from {AI_MODEL}...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=AI_MODEL,
            max_tokens=1800,
            messages=[{"role": "user", "content": "".join(lines)}],
        )
        return msg.content[0].text
    except Exception as e:
        print(f"[AI] Error: {e}")
        return None


def _md_to_html(text: str) -> str:
    """Minimal Markdown → HTML: bold, headers, paragraphs."""
    import html as _he, re
    t = _he.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"^#{1,3} (.+)$", r"<h3>\1</h3>", t, flags=re.MULTILINE)
    t = t.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{t}</p>"


def write_html(df: pd.DataFrame, config_str: str, failed: list,
               ai_text: str | None = None) -> None:
    """Write all scanner results to a self-contained dark-themed sortable HTML page."""
    import html as _he
    import datetime as _dt

    scores = df["score"].dropna()
    s_lo, s_hi = float(scores.min()), float(scores.max())

    MAIN_F = ["symbol","score","price","analyst_rating","analyst_target","analyst_upside",
              "exp_date","dte","strike","moneyness","exp_move_pct","vs_em",
              "bid","ask","spread","volume","be_bid","pct_be_bid","open_int",
              "iv_rank","iv","hv30","delta","theta","ret","ann_rtn","profit_prob","ma200_pct"]
    MAIN_H = ["Symbol","Score","Price~","Rating","Target","Upside%",
              "Exp Date","DTE","Strike","Mness%","EM%","vs EM%",
              "Bid","Ask","Spread","Vol","BE(Bid)","%BE","OI",
              "IVR","IV%","HV30%","Delta","θ/day","Ret%","AnnRtn%","PProb%","MA200%"]

    IV_F = ["symbol","price","analyst_rating","analyst_target","analyst_upside",
            "exp_date","strike","exp_move_pct","vs_em","bid","ask","spread",
            "volume","iv_rank","iv","hv30","ann_rtn","theta","profit_prob"]
    IV_H = ["Symbol","Price~","Rating","Target","Upside%",
            "Exp Date","Strike","EM%","vs EM%","Bid","Ask","Spread",
            "Vol","IVR","IV%","HV30%","AnnRtn%","θ/day","PProb%"]

    no_earn  = df["earnings_date"].eq("")
    high_ivr = df["iv_rank"] >= 50
    iv_gt_hv = df["iv"] > df["hv30"]

    parts: list[str] = [
        f'<h1>Short Put Scanner</h1>'
        f'<p class="meta">{_he.escape(config_str)}'
        f'<br>Generated {_dt.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>'
    ]

    if ai_text:
        parts.append(
            f'<h2>AI Analysis (top {AI_TOP_N} by score)</h2>'
            f'<div class="ai-box">{_md_to_html(ai_text)}</div>'
        )

    earn_df = df[df["earnings_date"].ne("")]
    if not earn_df.empty:
        parts.append("<h2>Earnings Within Expiry Window</h2>")
        parts.append(_html_table(earn_df.sort_values("score", ascending=False),
                                 MAIN_F, MAIN_H, "tbl-earn", s_lo, s_hi))

    parts.append("<h2>All Results</h2>")
    parts.append(_html_table(df.sort_values("ann_rtn", ascending=False),
                             MAIN_F, MAIN_H, "tbl-main", s_lo, s_hi))

    for title, mask, tid in [
        ("HIGH IVR + IV > HV30 — Best Premium Candidates",
         high_ivr &  iv_gt_hv & no_earn, "tbl-both"),
        ("HIGH IVR (>=50) Only",
         high_ivr & ~iv_gt_hv & no_earn, "tbl-ivr"),
        ("IV > HV30 Only",
        ~high_ivr &  iv_gt_hv & no_earn, "tbl-hv"),
    ]:
        parts.append(f"<h2>{_he.escape(title)}</h2>")
        parts.append(_html_table(df[mask].sort_values("score", ascending=False),
                                 IV_F, IV_H, tid, s_lo, s_hi))

    if failed:
        fs = ", ".join(_he.escape(s) for s in failed)
        parts.append(f'<p class="failed">[!] {len(failed)} tickers skipped (rate limited): {fs}</p>')

    page = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Short Put Scanner</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        + "".join(parts)
        + f"<script>{_JS}</script></body></html>"
    )

    with open(HTML_OUT, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Saved HTML  to {HTML_OUT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from cache import DB_PATH
    stats = CACHE.stats()
    total = sum(stats.values()) if stats else 0
    print(f"Cache DB : {DB_PATH}")
    print(f"Cache    : {dict(stats)}  ({total} entries total)")
    print("Loading universe...")
    all_tickers = get_tickers()
    tickers = (
        random.sample(all_tickers, min(SAMPLE_SIZE, len(all_tickers)))
        if SAMPLE_SIZE else all_tickers
    )
    print(f"Scanning {len(tickers)} tickers  |  profile={RISK_PROFILE.upper()}  "
          f"|  DTE {DTE_MIN}-{DTE_MAX}  |  |Δ| {DELTA_MIN}-{DELTA_MAX}  "
          f"|  strike≤{MAX_STRIKE}  |  vol≥{MIN_VOLUME}  spread≤{MAX_SPREAD_PCT:.0%}\n")

    rows: list[PutRow] = []
    failed: list[str] = []   # tickers that hit max rate-limit wait
    rate_errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_ticker, sym): sym for sym in tickers}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                r = fut.result()
                if r:
                    rows.append(r)
                rate_errors = 0
            except Exception as e:
                msg = str(e)
                if "Rate" in msg or "429" in msg:
                    rate_errors += 1
                    wait = min(5 * rate_errors, 60)
                    if wait >= 60:
                        print(f"[{sym}] rate limited — max wait reached, skipping")
                        failed.append(sym)
                        rate_errors = 0   # reset so next ticker gets a fresh chance
                    else:
                        print(f"[{sym}] rate limited — waiting {wait}s...")
                        time.sleep(wait)
                else:
                    print(f"[{sym}] error: {e!r}")

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.sort_values("ann_rtn", ascending=False).reset_index(drop=True)

    def _fmt_ratings(col_rating: pd.Series, col_num: pd.Series) -> pd.Series:
        abbrev = col_rating.map(_RATING_ABBREV).fillna(col_rating)
        has_n  = col_num > 0
        return abbrev.where(~has_n, abbrev + " (" + col_num.astype(str) + ")")

    display_map = {
        "symbol":          "Symbol",
        "score":           "Score",
        "price":           "Price~",
        "analyst_rating":  "Rating",
        "analyst_target":  "Target",
        "analyst_upside":  "Upside%",
        "exp_date":        "Exp Date",
        "dte":             "DTE",
        "strike":          "Strike",
        "moneyness":       "Mness%",
        "exp_move_pct":    "EM%",
        "vs_em":           "vs EM%",
        "bid":             "Bid",
        "ask":             "Ask",
        "spread":          "Spread",
        "volume":          "Vol",
        "be_bid":          "BE(Bid)",
        "pct_be_bid":      "%BE",
        "open_int":        "OI",
        "iv_rank":         "IVR",
        "iv":              "IV%",
        "hv30":            "HV30%",
        "delta":           "Delta",
        "theta":           "θ/day",
        "ret":             "Ret%",
        "ann_rtn":         "AnnRtn%",
        "profit_prob":     "PProb%",
        "ma200_pct":       "MA200%",
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

    # Earnings summary
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

    no_earn   = df["earnings_date"].eq("")
    high_ivr  = df["iv_rank"] >= 50
    iv_gt_hv  = df["iv"] > df["hv30"]

    _iv_section(df[ high_ivr & ~iv_gt_hv & no_earn].head(20),
                "HIGH IVR (>=50) only  — options elevated vs own history")
    _iv_section(df[~high_ivr &  iv_gt_hv & no_earn].head(20),
                "IV > HV30 only  — options priced above realized vol")
    _iv_section(df[ high_ivr &  iv_gt_hv & no_earn].head(20),
                "HIGH IVR + IV > HV30  — best premium candidates")

    df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved {len(df)} results to {CSV_OUT}")

    top_rows = sorted(rows, key=lambda r: r.score, reverse=True)[:AI_TOP_N]
    ai_text  = ai_analysis(top_rows)
    if ai_text:
        print(f"\n{'='*60}")
        print(f"AI ANALYSIS (top {len(top_rows)} by score)")
        print('='*60)
        print(ai_text)

    config_str = (f"Universe: {UNIVERSE}  |  Profile: {RISK_PROFILE}  |  "
                  f"DTE: {DTE_MIN}-{DTE_MAX}  |  |Δ|: {DELTA_MIN}-{DELTA_MAX}  |  "
                  f"strike≤{MAX_STRIKE}  |  vol≥{MIN_VOLUME}  |  spread≤{MAX_SPREAD_PCT:.0%}")
    write_html(df, config_str, failed, ai_text=ai_text)

    if failed:
        print(f"\n[!] {len(failed)} tickers skipped due to rate limiting:")
        print("    " + ", ".join(failed))
        print(f"    Re-run with UNIVERSE = {failed!r} to retry them.")


if __name__ == "__main__":
    main()
