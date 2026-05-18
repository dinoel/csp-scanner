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

import io
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

import analytics
import fundamentals as _fund
import sentiment as _sentiment
import skew as skewlib
from ai import ai_analysis
from cache import CACHE
from models import PutRow
from providers import get_provider
from report import RATING_ABBREV, write_html

# ── CONFIG ────────────────────────────────────────────────────────────────────
#UNIVERSE: str | list[str] = ["NVDA", "INTC", "AMD", "PLTR", "MCD", "ASTS", "TEAM", "NBIS", "IREN", "EMR", "ORCL", "DELL", "SMCI", "HAL", "GLW", "FCX", "HUT", "ARM"]
#UNIVERSE: str | list[str] = ["CRWV", "NVDA", "INTC", "MCD", "ASTS"]
UNIVERSE: str | list[str] = "screener"
SAMPLE_SIZE: int | None = None        # tickers to sample; None = full universe
RISK_PROFILE = "medium"              # "low" | "medium" | "high"
RISK_FREE_RATE = 0.05
MIN_BID = 0.05
COMPUTE_IV_RANK = True
MAX_WORKERS = 6      # parallel ticker scans; keep ≤8 to avoid Yahoo rate limits
CSV_OUT  = "csp_scan.csv"
HTML_OUT = "csp_scan.html"
ENABLE_AI_ANALYSIS = False
AI_TOP_N = 10
AI_MODEL = "claude-opus-4-7"
DATA_PROVIDER = "yfinance"  # "yfinance" | "massive" (set MASSIVE_API_KEY env var)
ENABLE_FUNDAMENTALS = False  # requires FINANCIALDATASETS_API_KEY env var
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

def _effective_price(row) -> tuple[float, float]:
    """Return (price_for_iv, bid_for_return) or (nan, nan) if liquidity filters fail."""
    oi = _int(row.get("openInterest"))
    if oi < MIN_OPEN_INTEREST:
        return float("nan"), float("nan")

    vol = _int(row.get("volume"))
    if vol < MIN_VOLUME:
        return float("nan"), float("nan")

    # Staleness check: yfinance 'volume' is from the last session the contract
    # actually traded — not necessarily today. Reject options whose last trade
    # predates the most recent valid trading session.
    last_trade = row.get("lastTradeDate")
    if last_trade is not None:
        try:
            trade_date = (last_trade.date() if hasattr(last_trade, "date")
                          else datetime.strptime(str(last_trade)[:10], "%Y-%m-%d").date())
            if trade_date < _last_valid_trade_date():
                return float("nan"), float("nan")
        except Exception:
            pass

    bid  = float(row.get("bid")       or 0)
    ask  = float(row.get("ask")       or 0)
    last = float(row.get("lastPrice") or 0)

    if bid > 0 and ask > 0:
        if (ask - bid) / bid > MAX_SPREAD_PCT:
            return float("nan"), float("nan")
        return (bid + ask) / 2, bid

    if bid > MIN_BID:
        return bid, bid

    if last > MIN_BID:
        if vol < max(MIN_VOLUME * 5, 50):
            return float("nan"), float("nan")
        return last, last

    return float("nan"), float("nan")


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
                  ma200: float, em: float) -> Optional[tuple]:
    """Score every eligible put strike and return the best one."""
    best = None
    best_score = float("-inf")
    dte = max(T * 365.0, 1.0)

    _need = [c for c in ("strike", "bid", "ask", "lastPrice", "volume",
                         "openInterest", "lastTradeDate")
             if c in puts.columns]
    for row in puts[_need].to_dict("records"):
        price_iv, bid_ret = _effective_price(row)
        if math.isnan(price_iv) or math.isnan(bid_ret):
            continue
        strike = float(row["strike"])
        iv = analytics.compute_iv(spot, strike, T, r, price_iv)
        if math.isnan(iv):
            continue
        delta = analytics.bs_put_delta(spot, strike, T, r, iv)
        if math.isnan(delta) or not (DELTA_MIN <= abs(delta) <= DELTA_MAX):
            continue
        if MAX_STRIKE and strike > MAX_STRIKE:
            continue

        pp    = analytics.calc_profit_prob(spot, strike, T, r, iv)
        ann   = (bid_ret / strike * 100.0) * (365.0 / dte)
        vs_em = (spot - strike) / em * 100.0 if (not math.isnan(em) and em > 0) else float("nan")
        ma_sc = analytics.ma200_score(spot, strike, ma200)
        sc    = score_put(ann, pp, vs_em, ma_sc)

        if sc > best_score:
            best_score = sc
            best = (strike, iv, bid_ret, float(row.get("ask") or 0),
                    _int(row.get("volume")), _int(row.get("openInterest")), delta)
    return best


# ── Universe loaders ──────────────────────────────────────────────────────────

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
    raise ValueError(f"Unknown universe: {UNIVERSE!r}")


# ── Per-ticker scan ───────────────────────────────────────────────────────────

def scan_ticker(symbol: str) -> Optional[PutRow]:
    from_cache = CACHE.get(f"fast_info:{symbol}") is not None
    print(f"[{symbol}] {'(cache) ' if from_cache else ''}fetching...")

    meta = CACHE.fetch(f"fast_info:{symbol}", lambda: PROVIDER.get_spot_and_expirations(symbol))
    spot = float(meta.get("price") or 0)
    if not spot or math.isnan(spot) or spot <= 0:
        return None

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
        print(f"[{symbol}] no expiry in {DTE_MIN}-{DTE_MAX} DTE")
        return None

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
            pp        = analytics.calc_profit_prob(spot, strike, T, RISK_FREE_RATE, iv)
            theta     = -analytics.bs_put_theta(spot, strike, T, RISK_FREE_RATE, iv)
            em_pct    = em / spot * 100.0 if not math.isnan(em) else float("nan")
            vs_em     = (spot - strike) / em * 100.0 if (not math.isnan(em) and em > 0) else float("nan")
            ivr       = analytics.get_iv_rank(hist, iv) if COMPUTE_IV_RANK else float("nan")
            ma_sc     = analytics.ma200_score(spot, strike, ma200)
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
            earn      = check_earnings(cal, today, expiry_dt)
            sc        = score_put(ann, pp, vs_em, ma_sc, fund.score)

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
                fundamental_score=fund.score,
                rev_growth=fund.rev_growth_pct,
                eps_beat_rate=fund.eps_beat_rate,
                fcf_margin=fund.fcf_margin_pct,
                bull_pct=float("nan"),  # fetched post-scan for top N only
                rsi=rsi,
                pcr=pcr,
                rr_25d_pct=skew_m.rr_25d_pct,
                score=_r(sc, 1),
            )

            if sc > best_score:
                best_score = sc
                best_row = row

        except Exception as e:
            print(f"[{symbol}] {expiry}: {e!r}")
            continue

    if best_row is None:
        print(f"[{symbol}] no qualifying put")
    return best_row


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
    failed: list[str] = []
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
                        rate_errors = 0
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

    df.to_csv(CSV_OUT, index=False)
    print(f"\nSaved {len(df)} results to {CSV_OUT}")

    top_rows = sorted(rows, key=lambda r: r.score, reverse=True)[:AI_TOP_N]

    fund_raw: dict[str, dict] = {}
    print(f"\nFetching sentiment + fundamentals for top {len(top_rows)} candidates...")
    for r in top_rows:
        # StockTwits sentiment (always, no key needed)
        sent = CACHE.fetch(f"sentiment:{r.symbol}",
                           lambda s=r.symbol: _sentiment.fetch_sentiment(s))
        r.bull_pct = sent.get("bull_pct", float("nan"))
        df.loc[df["symbol"] == r.symbol, "bull_pct"] = r.bull_pct

        # Fundamentals (only when API key is configured)
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

    ai_text = None
    if ENABLE_AI_ANALYSIS:
        ai_text = ai_analysis(top_rows, fund_raw, model=AI_MODEL)
    if ai_text:
        print(f"\n{'='*60}")
        print(f"AI ANALYSIS (top {len(top_rows)} by score)")
        print('='*60)
        print(ai_text)

    config_str = (f"Universe: {UNIVERSE}  |  Profile: {RISK_PROFILE}  |  "
                  f"DTE: {DTE_MIN}-{DTE_MAX}  |  |Δ|: {DELTA_MIN}-{DELTA_MAX}  |  "
                  f"strike≤{MAX_STRIKE}  |  vol≥{MIN_VOLUME}  |  spread≤{MAX_SPREAD_PCT:.0%}")
    write_html(df, config_str, failed, ai_text=ai_text, ai_top_n=AI_TOP_N, html_out=HTML_OUT)

    if failed:
        print(f"\n[!] {len(failed)} tickers skipped due to rate limiting:")
        print("    " + ", ".join(failed))
        print(f"    Re-run with UNIVERSE = {failed!r} to retry them.")


if __name__ == "__main__":
    main()
