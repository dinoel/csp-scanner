"""Generic options-trade primitives: Leg, TradeIdea, numerical EV engine."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import analytics


# ── Primitives ────────────────────────────────────────────────────────────────

@dataclass
class Leg:
    """Single options contract leg."""
    side:   str            # "long" | "short"
    kind:   str            # "call" | "put"
    strike: float
    qty:    int    = 1     # contracts; >1 for ratio spreads
    bid:    float  = 0.0
    ask:    float  = 0.0
    iv:     float  = 0.0   # leg's own IV (decimal, e.g. 0.45)
    delta:  float  = 0.0   # leg's own BS delta
    vol:    int    = 0
    oi:     int    = 0

    def cost_per_share(self, fill: str = "bid_ask") -> float:
        """Per-share entry cost. Positive = we pay, negative = we receive credit.

        `fill="bid_ask"` = worst case (short@bid, long@ask).
        `fill="mid"`     = mid-price both legs (realistic for liquid).
        """
        if fill == "mid":
            px = (self.bid + self.ask) / 2.0 if (self.bid > 0 and self.ask > 0) else (self.bid or self.ask)
        elif fill == "bid_ask":
            px = self.bid if self.side == "short" else self.ask
        else:
            raise ValueError(f"unknown fill {fill!r}")
        sign = -1 if self.side == "short" else 1
        return sign * px * self.qty

    def payoff_per_share(self, S: float) -> float:
        """Per-share intrinsic payoff at expiry given spot S (excludes premium)."""
        if self.kind == "put":
            intrinsic = max(self.strike - S, 0.0)
        elif self.kind == "call":
            intrinsic = max(S - self.strike, 0.0)
        else:
            raise ValueError(f"unknown kind {self.kind!r}")
        sign = -1 if self.side == "short" else 1
        return sign * intrinsic * self.qty


@dataclass
class TradeIdea:
    """One candidate trade — built from legs + market context."""
    symbol:    str
    expiry:    str
    dte:       int
    strategy:  str           # "csp" | "bps" | "jade_lizard" | ...
    legs:      list[Leg]
    spot:      float
    iv_ref:    float = float("nan")   # representative IV (e.g. short-leg IV)
    hv30:      float = float("nan")   # decimal, e.g. 0.45

    # ── Computed by analyze() ────────────────────────────────────────────────
    credit:        float = 0.0       # net credit per contract @ bid_ask  ($, positive = received)
    credit_mid:    float = 0.0       # net credit per contract @ mid       ($)
    max_gain:      float = 0.0       # per-contract max profit ($)
    max_loss:      float = 0.0       # per-contract max loss as a positive number ($)
                                      # NaN/inf if undefined (e.g. naked short calls)
    breakevens:    list[float] = field(default_factory=list)
    p_profit:      float = 0.0       # P(payoff > 0) integrated with IV  (%)
    p_profit_hv30: float = 0.0       # same with HV30                       (%)
    ev:            float = 0.0       # numerical EV under IV, bid_ask fill ($/contract)
    ev_mid:        float = 0.0       # numerical EV under IV, mid fill
    ev_hv30:       float = 0.0       # numerical EV under HV30, bid_ask
    ev_managed:    float = 0.0       # 50%-profit early exit at half DTE
    roi_ann:       float = 0.0       # primary ranking metric: ev_mid / max_loss × 365/dte × 100

    # ── Pretty leg label for console / UI ────────────────────────────────────
    def leg_label(self) -> str:
        """Compact one-line representation: '-1P184/+1P183' style."""
        parts = []
        for L in self.legs:
            sign = "-" if L.side == "short" else "+"
            k    = "P" if L.kind == "put" else "C"
            q    = f"{L.qty}×" if L.qty != 1 else ""
            parts.append(f"{sign}{q}{k}{int(L.strike) if L.strike == int(L.strike) else L.strike:g}")
        return " ".join(parts)


# ── Numerical EV engine ───────────────────────────────────────────────────────

def numerical_ev(legs: list[Leg], spot: float, T: float, r: float, sigma: float,
                 fill: str = "bid_ask", n_points: int = 400) -> float:
    """E[total payoff per contract], in $, under risk-neutral lognormal pricing.

    payoff_total(S) = Σ leg.payoff_per_share(S) − Σ leg.cost_per_share(fill)
    EV = ∫ payoff_total(S) × pdf(S) dS  × 100 (per-contract)
    """
    if T <= 0 or sigma <= 0 or math.isnan(sigma) or spot <= 0:
        return float("nan")

    log_spot = math.log(spot)
    drift    = (r - 0.5 * sigma * sigma) * T
    std      = sigma * math.sqrt(T)
    # ±5 sigma in log space — covers practical tail
    lo, hi   = log_spot + drift - 5 * std, log_spot + drift + 5 * std
    grid_log = np.linspace(lo, hi, n_points)
    grid_S   = np.exp(grid_log)

    # payoff per share at each spot value
    payoff_intrinsic = np.array([sum(L.payoff_per_share(S) for L in legs) for S in grid_S])
    net_cost         = sum(L.cost_per_share(fill) for L in legs)
    payoff_total     = payoff_intrinsic - net_cost   # entry cost subtracted once

    # lognormal pdf in S-space
    log_diff = grid_log - (log_spot + drift)
    pdf      = np.exp(-0.5 * (log_diff / std) ** 2) / (grid_S * std * math.sqrt(2 * math.pi))

    ev_per_share = np.trapezoid(payoff_total * pdf, grid_S)
    return ev_per_share * 100.0


def probability_of_profit(legs: list[Leg], spot: float, T: float, r: float,
                          sigma: float, fill: str = "bid_ask",
                          n_points: int = 400) -> float:
    """P(total P/L at expiry > 0), in %, via the same integration grid."""
    if T <= 0 or sigma <= 0 or math.isnan(sigma) or spot <= 0:
        return float("nan")
    log_spot = math.log(spot)
    drift    = (r - 0.5 * sigma * sigma) * T
    std      = sigma * math.sqrt(T)
    lo, hi   = log_spot + drift - 5 * std, log_spot + drift + 5 * std
    grid_log = np.linspace(lo, hi, n_points)
    grid_S   = np.exp(grid_log)
    payoff   = np.array([sum(L.payoff_per_share(S) for L in legs) for S in grid_S])
    net_cost = sum(L.cost_per_share(fill) for L in legs)
    pnl      = payoff - net_cost
    log_diff = grid_log - (log_spot + drift)
    pdf      = np.exp(-0.5 * (log_diff / std) ** 2) / (grid_S * std * math.sqrt(2 * math.pi))
    mask     = pnl > 0
    return float(np.trapezoid(pdf[mask], grid_S[mask]) * 100.0) if mask.any() else 0.0


def find_breakevens(legs: list[Leg], spot: float, fill: str = "bid_ask",
                    span: float = 0.6, n_points: int = 600) -> list[float]:
    """Find spot prices where total P/L = 0 by sign-change detection.

    Strike points are inserted into the grid so linear interpolation crosses
    payoff kinks exactly (otherwise BE can be off by half a grid step).
    """
    lo, hi   = spot * (1 - span), spot * (1 + span)
    grid     = sorted(set(np.linspace(lo, hi, n_points).tolist()
                          + [L.strike       for L in legs]
                          + [L.strike - 1e-4 for L in legs]
                          + [L.strike + 1e-4 for L in legs]))
    net_cost = sum(L.cost_per_share(fill) for L in legs)
    pnl      = [sum(L.payoff_per_share(S) for L in legs) - net_cost for S in grid]
    bes: list[float] = []
    for i in range(len(pnl) - 1):
        pa, pb = pnl[i], pnl[i + 1]
        if pa == 0:
            bes.append(round(grid[i], 2))
            continue
        if pa * pb < 0:
            a, b = grid[i], grid[i + 1]
            be   = a - pa * (b - a) / (pb - pa)
            bes.append(round(be, 2))
    # dedupe (kink boundaries can produce micro-duplicates)
    return sorted(set(bes))


def max_gain_loss(legs: list[Leg], spot: float, fill: str = "bid_ask") -> tuple[float, float]:
    """Per-contract (max_gain, max_loss). max_loss returned as positive number.

    Sweeps a wide grid plus the strike points to find extrema. Returns
    (NaN, inf) if loss is unbounded (naked short calls etc.).
    """
    strikes  = sorted({L.strike for L in legs})
    if not strikes:
        return float("nan"), float("nan")
    lo, hi   = max(0.01, min(strikes) * 0.5), max(strikes) * 1.5
    grid     = sorted(set(np.linspace(lo, hi, 200).tolist() + strikes + [spot]))
    net_cost = sum(L.cost_per_share(fill) for L in legs)
    pnls     = [sum(L.payoff_per_share(S) for L in legs) - net_cost for S in grid]
    # Loss unbounded check: extrapolate to S=0 and S→∞
    pnl_zero  = sum(L.payoff_per_share(0)  for L in legs) - net_cost
    pnl_huge  = sum(L.payoff_per_share(hi * 100) for L in legs) - net_cost
    candidates_pnl = pnls + [pnl_zero, pnl_huge]
    max_g = max(candidates_pnl) * 100.0
    min_p = min(candidates_pnl) * 100.0
    max_l = -min_p  # positive number
    # Check if loss diverges (naked short side)
    pnl_huger = sum(L.payoff_per_share(hi * 1000) for L in legs) - net_cost
    if pnl_huger * 100.0 < min_p - 1.0:
        max_l = float("inf")
    return max_g, max_l


# ── Analysis: take a list of legs, compute every metric on a TradeIdea ────────

def analyze(idea: TradeIdea, *, r: float = 0.05) -> TradeIdea:
    """Populate all computed metrics on `idea` in-place. Returns the same idea."""
    T = idea.dte / 365.0
    spot = idea.spot
    sigma_iv = idea.iv_ref
    sigma_hv = idea.hv30

    # Pricing — bid/ask vs mid
    idea.credit     = -sum(L.cost_per_share("bid_ask") for L in idea.legs) * 100.0
    idea.credit_mid = -sum(L.cost_per_share("mid")     for L in idea.legs) * 100.0

    # Max gain/loss (defined-risk → bounded; naked → inf)
    idea.max_gain, idea.max_loss = max_gain_loss(idea.legs, spot, fill="bid_ask")

    # Breakevens
    idea.breakevens = find_breakevens(idea.legs, spot, fill="bid_ask")

    # P(profit) and EVs
    idea.p_profit      = probability_of_profit(idea.legs, spot, T, r, sigma_iv, fill="bid_ask")
    idea.p_profit_hv30 = probability_of_profit(idea.legs, spot, T, r, sigma_hv, fill="bid_ask") if not math.isnan(sigma_hv) else float("nan")
    idea.ev            = numerical_ev(idea.legs, spot, T, r, sigma_iv, fill="bid_ask")
    idea.ev_mid        = numerical_ev(idea.legs, spot, T, r, sigma_iv, fill="mid")
    idea.ev_hv30       = numerical_ev(idea.legs, spot, T, r, sigma_hv, fill="bid_ask") if not math.isnan(sigma_hv) else float("nan")

    # Managed: EV at T/2 with half-credit-gain target (approximation)
    T_half = T / 2.0
    if T_half > 0:
        ev_half = numerical_ev(idea.legs, spot, T_half, r, sigma_iv, fill="bid_ask")
        # Heuristic: if held to T/2, on average we'd capture ~half the credit
        # if profitable, full max_loss if not. Use ev_half as a proxy + half-credit cap.
        # Simple model: managed_ev = max(ev_half, -max_loss) but capped at 0.5×credit
        idea.ev_managed = max(min(ev_half, idea.credit * 0.5), -idea.max_loss) if not math.isnan(ev_half) else float("nan")
    else:
        idea.ev_managed = float("nan")

    # Annualized ROI on margin (primary ranking metric)
    if idea.max_loss > 0 and not math.isinf(idea.max_loss) and idea.dte > 0 and not math.isnan(idea.ev_mid):
        idea.roi_ann = idea.ev_mid / idea.max_loss * (365.0 / idea.dte) * 100.0
    else:
        idea.roi_ann = float("nan")

    return idea


# ── Helpers used by individual scanners (in strategies/scanners.py) ───────────

def find_candidate_legs(chain_df, spot: float, T: float, r: float,
                        kind: str,
                        eff_price_fn,
                        delta_min: float = 0.0, delta_max: float = 1.0,
                        max_strike: Optional[float] = None) -> list[Leg]:
    """Filter a chain (calls or puts DataFrame) into viable Leg objects.

    `eff_price_fn(row, oi_unavailable=False) -> (price_iv, bid_for_ret, reason)`
    is the existing csp_scanner._effective_price.
    """
    if chain_df is None or len(chain_df) == 0:
        return []

    oi_unavailable = (
        "openInterest" in chain_df.columns
        and float(chain_df["openInterest"].fillna(0).max() or 0) == 0
    )
    cols = [c for c in ("strike", "bid", "ask", "lastPrice", "volume",
                        "openInterest", "lastTradeDate") if c in chain_df.columns]
    rows = chain_df[cols].to_dict("records")
    legs: list[Leg] = []
    for row in rows:
        price_iv, _, reason = eff_price_fn(row, oi_unavailable=oi_unavailable)
        if reason:
            continue
        strike = float(row["strike"])
        if max_strike and strike > max_strike:
            continue
        if kind == "put":
            iv = analytics.compute_iv(spot, strike, T, r, price_iv)
            delta = analytics.bs_put_delta(spot, strike, T, r, iv) if not math.isnan(iv) else float("nan")
        else:  # call
            iv = analytics.compute_iv_call(spot, strike, T, r, price_iv)
            delta = analytics.bs_call_delta(spot, strike, T, r, iv) if not math.isnan(iv) else float("nan")
        if math.isnan(iv) or math.isnan(delta):
            continue
        if not (delta_min <= abs(delta) <= delta_max):
            continue
        legs.append(Leg(
            side="short",          # default; scanner flips as needed
            kind=kind,
            strike=strike,
            qty=1,
            bid=float(row.get("bid") or 0),
            ask=float(row.get("ask") or 0),
            iv=iv,
            delta=delta,
            vol=int(float(row.get("volume") or 0) if not pd_isna(row.get("volume")) else 0),
            oi=int(float(row.get("openInterest") or 0) if not pd_isna(row.get("openInterest")) else 0),
        ))
    legs.sort(key=lambda L: L.strike)
    return legs


def pd_isna(v) -> bool:
    """math.isnan-safe NaN detection that handles None and non-floats."""
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False
