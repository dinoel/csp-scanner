"""Volatility skew computation — shared between csp_scanner and skew_scanner.

Primary metric: 25-delta Risk Reversal
    rr_25d     = IV(put_25Δ) − IV(call_25Δ)   [in vol points]
    rr_25d_pct = rr_25d / ATM_IV × 100         [normalised, cross-name comparable]

Interpretation for a put SELLER:
    rr_25d_pct > 0   puts costlier than calls — normal fear skew
    rr_25d_pct > 15  steep skew — rich premium but market pricing real downside risk
    rr_25d_pct < 0   calls costlier than puts — unusual bullish frenzy / M&A squeeze
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

_NAN = float("nan")


@dataclass
class SkewMetrics:
    atm_iv:     float   # ATM implied vol (from nearest call, annualised)
    rr_25d:     float   # 25Δ risk reversal in vol points
    rr_25d_pct: float   # normalised RR  (%)
    rr_10d:     float   # 10Δ risk reversal in vol points (NaN if unavailable)


EMPTY = SkewMetrics(atm_iv=_NAN, rr_25d=_NAN, rr_25d_pct=_NAN, rr_10d=_NAN)


# ── General BS helpers (call + put) ──────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S, K, T, r, sigma) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_put: bool) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return _NAN
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    if is_put:
        return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_put: bool) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return _NAN
    nd1 = _ncdf(_d1(S, K, T, r, sigma))
    return nd1 - 1.0 if is_put else nd1


def compute_iv(S: float, K: float, T: float, r: float, price: float, is_put: bool) -> float:
    """BS IV via bisection. Returns NaN if price is outside arbitrage bounds."""
    intrinsic = max(0.0, (K - S if is_put else S - K) * math.exp(-r * T))
    if price <= intrinsic or price <= 0:
        return _NAN
    lo, hi = 1e-4, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, r, mid, is_put) > price:
            hi = mid
        else:
            lo = mid
    iv = (lo + hi) / 2
    return iv if 0.01 < iv < 5.0 else _NAN


# ── Chain helpers ─────────────────────────────────────────────────────────────

def _mid_price(row) -> float:
    """Mid price if bid/ask live, else last price, else NaN. No liquidity filters."""
    bid  = float(row.get("bid")       or 0)
    ask  = float(row.get("ask")       or 0)
    last = float(row.get("lastPrice") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return last if last > 0 else _NAN


def find_at_delta(df: pd.DataFrame, spot: float, T: float, r: float,
                  target_delta: float, is_put: bool) -> tuple:
    """Return (strike, iv, bid, ask, last) of option whose |delta| ≈ target_delta.

    target_delta should be positive (e.g. 0.25 for 25-delta).
    """
    best = (None, _NAN, _NAN, _NAN, _NAN)
    best_dist = float("inf")
    target_signed = -target_delta if is_put else target_delta

    for row in df.to_dict("records"):
        price = _mid_price(row)
        if math.isnan(price):
            continue
        iv = compute_iv(spot, row["strike"], T, r, price, is_put)
        if math.isnan(iv):
            continue
        delta = bs_delta(spot, row["strike"], T, r, iv, is_put)
        if math.isnan(delta):
            continue
        dist = abs(delta - target_signed)
        if dist < best_dist:
            best_dist = dist
            best = (row["strike"], iv,
                    row.get("bid"), row.get("ask"), row.get("lastPrice"))
    return best


def find_atm_iv(calls: pd.DataFrame, spot: float, T: float, r: float) -> float:
    """ATM IV from the call closest to spot."""
    best_iv, best_dist = _NAN, float("inf")
    for row in calls.to_dict("records"):
        price = _mid_price(row)
        if math.isnan(price):
            continue
        iv = compute_iv(spot, row["strike"], T, r, price, is_put=False)
        if math.isnan(iv):
            continue
        dist = abs(row["strike"] - spot)
        if dist < best_dist:
            best_dist = dist
            best_iv = iv
    return best_iv


# ── Public API ────────────────────────────────────────────────────────────────

def compute_skew(calls: pd.DataFrame, puts: pd.DataFrame,
                 spot: float, T: float, r: float) -> SkewMetrics:
    """Compute vol skew metrics from an option chain."""
    if calls.empty or puts.empty:
        return EMPTY

    atm_iv = find_atm_iv(calls, spot, T, r)
    _, p25_iv, *_ = find_at_delta(puts,  spot, T, r, 0.25, True)
    _, c25_iv, *_ = find_at_delta(calls, spot, T, r, 0.25, False)
    _, p10_iv, *_ = find_at_delta(puts,  spot, T, r, 0.10, True)
    _, c10_iv, *_ = find_at_delta(calls, spot, T, r, 0.10, False)

    if math.isnan(atm_iv) or math.isnan(p25_iv) or math.isnan(c25_iv):
        return SkewMetrics(atm_iv=atm_iv, rr_25d=_NAN, rr_25d_pct=_NAN, rr_10d=_NAN)

    rr25 = p25_iv - c25_iv
    rr10 = (p10_iv - c10_iv) if not (math.isnan(p10_iv) or math.isnan(c10_iv)) else _NAN

    return SkewMetrics(
        atm_iv=round(atm_iv, 4),
        rr_25d=round(rr25, 4),
        rr_25d_pct=round(rr25 / atm_iv * 100.0, 2),
        rr_10d=round(rr10, 4) if not math.isnan(rr10) else _NAN,
    )
