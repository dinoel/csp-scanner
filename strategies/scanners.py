"""Per-strategy scanners. Each enumerates viable strike combinations and
returns `list[TradeIdea]`. Generic enough that adding a new strategy is
~30 lines: pick legs, build a TradeIdea, call analyze().
"""
from __future__ import annotations

import math
from dataclasses import replace

from .base import Leg, TradeIdea, analyze


# ── Single-leg ────────────────────────────────────────────────────────────────

def scan_csps(viable_puts: list[Leg], *,
              symbol: str, expiry: str, dte: int, spot: float,
              iv_ref_lookup, hv30: float,
              delta_min: float, delta_max: float, r: float = 0.05) -> list[TradeIdea]:
    """Cash-secured put: single short put per strike in delta window."""
    ideas: list[TradeIdea] = []
    for leg_template in viable_puts:
        if not (delta_min <= abs(leg_template.delta) <= delta_max):
            continue
        short = replace(leg_template, side="short", qty=1)
        idea = TradeIdea(
            symbol=symbol, expiry=expiry, dte=dte, strategy="csp",
            legs=[short], spot=spot, iv_ref=short.iv, hv30=hv30,
        )
        analyze(idea, r=r)
        ideas.append(idea)
    return ideas


# ── Two-leg verticals ─────────────────────────────────────────────────────────

def scan_bps(viable_puts: list[Leg], *,
             symbol: str, expiry: str, dte: int, spot: float,
             hv30: float, delta_min: float, delta_max: float,
             max_width: float = 20.0, min_credit: float = 0.05,
             r: float = 0.05) -> list[TradeIdea]:
    """Bull put spread: sell put A + buy put B, B < A."""
    ideas: list[TradeIdea] = []
    for i, short_tmpl in enumerate(viable_puts):
        if not (delta_min <= abs(short_tmpl.delta) <= delta_max):
            continue
        for j in range(i - 1, -1, -1):
            long_tmpl = viable_puts[j]
            width = short_tmpl.strike - long_tmpl.strike
            if width <= 0:
                continue
            if width > max_width:
                break
            if long_tmpl.ask <= 0:
                continue
            credit = short_tmpl.bid - long_tmpl.ask
            if credit < min_credit:
                continue
            short = replace(short_tmpl, side="short", qty=1)
            long_ = replace(long_tmpl,  side="long",  qty=1)
            idea = TradeIdea(
                symbol=symbol, expiry=expiry, dte=dte, strategy="bps",
                legs=[short, long_], spot=spot, iv_ref=short.iv, hv30=hv30,
            )
            analyze(idea, r=r)
            ideas.append(idea)
    return ideas


def scan_bcs(viable_calls: list[Leg], *,
             symbol: str, expiry: str, dte: int, spot: float,
             hv30: float, delta_min: float, delta_max: float,
             max_width: float = 20.0, min_credit: float = 0.05,
             r: float = 0.05) -> list[TradeIdea]:
    """Bear call spread: sell call A + buy call B, B > A. Mirror of BPS."""
    ideas: list[TradeIdea] = []
    for i, short_tmpl in enumerate(viable_calls):
        if not (delta_min <= short_tmpl.delta <= delta_max):
            continue
        for j in range(i + 1, len(viable_calls)):
            long_tmpl = viable_calls[j]
            width = long_tmpl.strike - short_tmpl.strike
            if width <= 0:
                continue
            if width > max_width:
                break
            if long_tmpl.ask <= 0:
                continue
            credit = short_tmpl.bid - long_tmpl.ask
            if credit < min_credit:
                continue
            short = replace(short_tmpl, side="short", qty=1)
            long_ = replace(long_tmpl,  side="long",  qty=1)
            idea = TradeIdea(
                symbol=symbol, expiry=expiry, dte=dte, strategy="bcs",
                legs=[short, long_], spot=spot, iv_ref=short.iv, hv30=hv30,
            )
            analyze(idea, r=r)
            ideas.append(idea)
    return ideas


# ── Exotic: Jade Lizard ───────────────────────────────────────────────────────

def scan_jade_lizards(viable_puts: list[Leg], viable_calls: list[Leg], *,
                      symbol: str, expiry: str, dte: int, spot: float,
                      hv30: float, delta_min: float, delta_max: float,
                      max_width: float = 20.0, min_credit: float = 0.05,
                      r: float = 0.05) -> list[TradeIdea]:
    """Jade Lizard = short put + short bear call spread.

    Magic feature: when total credit >= width of the call spread, the trade
    has NO upside risk. Profit zone covers the entire upside.
    """
    ideas: list[TradeIdea] = []

    short_puts = [p for p in viable_puts if delta_min <= abs(p.delta) <= delta_max]

    # Prune: top-K BCS pairs per expiry to limit O(N×M²) explosion.
    # Build viable BCS pairs and keep the top by credit/width ratio.
    bcs_pairs: list[tuple[Leg, Leg, float]] = []  # (short_call, long_call, width)
    for i, sc in enumerate(viable_calls):
        if not (delta_min <= sc.delta <= delta_max):
            continue
        for j in range(i + 1, len(viable_calls)):
            lc = viable_calls[j]
            w = lc.strike - sc.strike
            if w <= 0:
                continue
            if w > max_width:
                break
            if lc.ask <= 0:
                continue
            bcs_credit = sc.bid - lc.ask
            if bcs_credit < min_credit:
                continue
            bcs_pairs.append((sc, lc, w))
    # Keep top-K by credit/width (best premium per unit risk)
    bcs_pairs.sort(key=lambda t: (t[0].bid - t[1].ask) / t[2], reverse=True)
    bcs_pairs = bcs_pairs[:6]

    for sp_tmpl in short_puts:
        for sc_tmpl, lc_tmpl, width in bcs_pairs:
            # Build all three legs
            sp = replace(sp_tmpl, side="short", qty=1)
            sc = replace(sc_tmpl, side="short", qty=1)
            lc = replace(lc_tmpl, side="long",  qty=1)
            # Total credit ≥ 0 sanity check (will be redone in analyze)
            total_credit_per_share = sp.bid + sc.bid - lc.ask
            if total_credit_per_share < min_credit:
                continue
            idea = TradeIdea(
                symbol=symbol, expiry=expiry, dte=dte, strategy="jade_lizard",
                legs=[sp, sc, lc], spot=spot, iv_ref=sp.iv, hv30=hv30,
            )
            analyze(idea, r=r)
            ideas.append(idea)
    return ideas


# ── Cross-strategy orchestrator ───────────────────────────────────────────────

def scan_all(viable_puts: list[Leg], viable_calls: list[Leg], *,
             symbol: str, expiry: str, dte: int, spot: float,
             hv30: float, delta_min: float, delta_max: float,
             max_width: float = 20.0, min_credit: float = 0.05,
             r: float = 0.05) -> list[TradeIdea]:
    """Run every implemented scanner and return all ideas for one expiry."""
    out: list[TradeIdea] = []
    out += scan_csps(viable_puts, symbol=symbol, expiry=expiry, dte=dte, spot=spot,
                     iv_ref_lookup=None, hv30=hv30,
                     delta_min=delta_min, delta_max=delta_max, r=r)
    out += scan_bps(viable_puts, symbol=symbol, expiry=expiry, dte=dte, spot=spot,
                    hv30=hv30, delta_min=delta_min, delta_max=delta_max,
                    max_width=max_width, min_credit=min_credit, r=r)
    out += scan_bcs(viable_calls, symbol=symbol, expiry=expiry, dte=dte, spot=spot,
                    hv30=hv30, delta_min=delta_min, delta_max=delta_max,
                    max_width=max_width, min_credit=min_credit, r=r)
    out += scan_jade_lizards(viable_puts, viable_calls,
                             symbol=symbol, expiry=expiry, dte=dte, spot=spot,
                             hv30=hv30, delta_min=delta_min, delta_max=delta_max,
                             max_width=max_width, min_credit=min_credit, r=r)
    return out
