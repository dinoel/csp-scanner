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
#UNIVERSE: str | list[str] = ["NVDA", "INTC", "AMD", "PLTR", "MCD", "ASTS", "TEAM", "NBIS", "IREN", "EMR", "ORCL", "DELL", "SMCI", "HAL", "GLW", "FCX"]
UNIVERSE: str | list[str] = "sp500"
SAMPLE_SIZE: int | None = 101        # tickers to sample; None = full universe
RISK_PROFILE = "medium"              # "low" | "medium" | "high"
RISK_FREE_RATE = 0.05
MIN_BID = 0.05
COMPUTE_IV_RANK = True
MAX_WORKERS = 6      # parallel ticker scans; keep ≤8 to avoid Yahoo rate limits
CSV_OUT = "csp_scan.csv"

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
        MIN_OPEN_INTEREST=50,  MIN_VOLUME=500,  MAX_SPREAD_PCT=0.50,
        SCORE_W_PROB=0.45, SCORE_W_RETURN=0.25, SCORE_W_SAFETY=0.20, SCORE_W_MA200=0.10,
    ),
    "high": dict(
        DTE_MIN=7,   DTE_MAX=31,           # shorter DTE → higher annualized return
        DELTA_MIN=0.15, DELTA_MAX=0.50,    # closer ATM — more premium, more risk
        MAX_STRIKE=200,
        MIN_OPEN_INTEREST=50,  MIN_VOLUME=10,  MAX_SPREAD_PCT=0.60,
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
    delta: float          # negative (put delta)
    theta: float          # daily $ gain for seller (positive)
    ret: float            # bid/strike*100  (return on capital %)
    ann_rtn: float        # annualized return %
    profit_prob: float    # P(expires OTM) %
    earnings_date: str        # earnings date 'YYYY-MM-DD' if in window, else ''
    ma200_pct: float      # (spot - MA200) / MA200 * 100; positive = above MA200
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
    ma200     = get_ma200(hist)
    ma200_pct = _r((spot - ma200) / ma200 * 100.0) if not math.isnan(ma200) else float("nan")

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
                delta=_r(delta, 3),
                theta=_r(theta, 3),
                ret=_r(ret),
                ann_rtn=_r(ann, 1),
                profit_prob=_r(pp, 1),
                earnings_date=earn or "",
                ma200_pct=ma200_pct,
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
    raise ValueError(f"Unknown universe: {UNIVERSE!r}")


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
    rate_errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scan_ticker, sym): sym for sym in tickers}
        rate_errors = 0
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
                    print(f"[{sym}] rate limited — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[{sym}] error: {e!r}")

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.sort_values("ann_rtn", ascending=False).reset_index(drop=True)

    display_map = {
        "symbol":       "Symbol",
        "score":        "Score",
        "price":        "Price~",
        "exp_date":     "Exp Date",
        "dte":          "DTE",
        "strike":       "Strike",
        "moneyness":    "Mness%",
        "exp_move_pct": "EM%",
        "vs_em":        "vs EM%",
        "bid":          "Bid",
        "ask":          "Ask",
        "spread":       "Spread",
        "volume":       "Vol",
        "be_bid":       "BE(Bid)",
        "pct_be_bid":   "%BE",
        "open_int":     "OI",
        "iv_rank":      "IVR",
        "iv":           "IV%",
        "delta":        "Delta",
        "theta":        "θ/day",
        "ret":          "Ret%",
        "ann_rtn":      "AnnRtn%",
        "profit_prob":  "PProb%",
        "ma200_pct":    "MA200%",
    }

    disp = df[list(display_map)].rename(columns=display_map).copy()
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

    # High-IV-rank highlights (no earnings risk)
    high_iv = df[(df["iv_rank"] >= 50) & df["earnings_date"].eq("")].head(20)
    if not high_iv.empty:
        cols = ["symbol", "price", "exp_date", "strike", "exp_move_pct", "vs_em", "bid", "ask", "spread", "volume", "iv_rank", "iv", "ann_rtn", "theta", "profit_prob"]
        hdrs = ["Symbol", "Price~", "Exp Date", "Strike", "EM%", "vs EM%", "Bid", "Ask", "Spread", "Vol", "IVR", "IV%", "AnnRtn%", "θ/day", "PProb%"]
        print(f"\n--- HIGH IV RANK (>=50, no earnings risk) ---")
        print(high_iv[cols].rename(columns=dict(zip(cols, hdrs))).to_string(index=False))

    df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved {len(df)} results to {CSV_OUT}")


if __name__ == "__main__":
    main()
