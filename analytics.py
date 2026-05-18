"""Options analytics — Black-Scholes, implied vol, historical vol, scoring helpers.

All functions are pure (no config dependencies) and safe to import anywhere.
"""
from __future__ import annotations

import math
from datetime import datetime, date

import numpy as np
import pandas as pd


# ── Black-Scholes ─────────────────────────────────────────────────────────────

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


# ── Historical vol & IV rank ──────────────────────────────────────────────────

def get_ma200(hist: pd.DataFrame) -> float:
    """200-day simple moving average of closing price."""
    try:
        if hist is None or len(hist) < 200:
            return float("nan")
        return float(hist["Close"].squeeze().rolling(200).mean().iloc[-1])
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


def get_iv_rank(hist: pd.DataFrame, current_iv: float) -> float:
    """IV rank: where current ATM IV sits in 52-week rolling-HV range (0–100).
    Uses 21-day rolling historical volatility as a proxy for IV history."""
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


# ── Expected move ─────────────────────────────────────────────────────────────

def _mid_or_last(row) -> float:
    """Best available price: mid > bid > lastPrice."""
    bid  = float(row.get("bid")       or 0)
    ask  = float(row.get("ask")       or 0)
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


# ── Expiry picker ────────────────────────────────────────────────────────────

def pick_expiry(expirations: list[str], today: date, dte_min: int, dte_max: int):
    """Return (expiry_str, dte) for the nearest expiry inside [dte_min, dte_max]."""
    candidates = []
    for s in expirations:
        try:
            d   = datetime.strptime(s, "%Y-%m-%d").date()
            dte = (d - today).days
            if dte_min <= dte <= dte_max:
                candidates.append((dte, s))
        except ValueError:
            continue
    if not candidates:
        return None, None
    dte, s = min(candidates)
    return s, dte


# ── RSI ──────────────────────────────────────────────────────────────────────

def compute_rsi(hist: pd.DataFrame, period: int = 14) -> float:
    """Wilder's RSI from daily closing prices. Returns 0-100 or NaN."""
    try:
        if hist is None or len(hist) < period + 1:
            return float("nan")
        close = hist["Close"].squeeze()
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss
        return round(float(100 - (100 / (1 + rs)).iloc[-1]), 1)
    except Exception:
        return float("nan")


# ── MA200 sub-score ───────────────────────────────────────────────────────────

def ma200_score(spot: float, strike: float, ma200: float) -> float:
    """0-100 score for the MA200 component.

    Rewards uptrend (spot > MA200) and extra safety buffer
    when the strike sits below MA200 (the 200-day acts as support).
    """
    if math.isnan(ma200) or ma200 <= 0:
        return 50.0
    pct = (spot - ma200) / ma200 * 100.0
    if   pct >= 10: trend = 90
    elif pct >=  5: trend = 75
    elif pct >=  0: trend = 60
    elif pct >= -5: trend = 35
    elif pct >= -10: trend = 20
    else:           trend = 10
    bonus = 15 if (spot > ma200 and strike < ma200) else 0
    return min(float(trend + bonus), 100.0)
