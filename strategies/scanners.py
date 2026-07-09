"""Per-strategy scanners. Each enumerates viable strike combinations and
returns `list[TradeIdea]`. Generic enough that adding a new strategy is
~30 lines: pick legs, build a TradeIdea, call analyze().
"""
from __future__ import annotations

import math
from dataclasses import replace

from .base import Leg, TradeIdea, analyze


# ── Shared helper: enumerate vertical spread pairs ────────────────────────────

def _build_vertical_pairs(viable: list[Leg], *,
                          is_bull: bool,
                          delta_min: float, delta_max: float,
                          max_width: float, min_credit: float
                          ) -> list[tuple[Leg, Leg, float]]:
    """Return every viable (short, long, width) vertical pair.

    is_bull=True  → bull put spread pattern: long strike below short.
    is_bull=False → bear call spread pattern: long strike above short.
    Delta filter applies to the short leg only (long leg auto-picked below/above).
    """
    pairs: list[tuple[Leg, Leg, float]] = []
    for i, short_tmpl in enumerate(viable):
        # Delta target on short leg
        d = abs(short_tmpl.delta) if is_bull else short_tmpl.delta
        if not (delta_min <= d <= delta_max):
            continue
        candidates = range(i - 1, -1, -1) if is_bull else range(i + 1, len(viable))
        for j in candidates:
            long_tmpl = viable[j]
            width = (short_tmpl.strike - long_tmpl.strike) if is_bull \
                    else (long_tmpl.strike - short_tmpl.strike)
            if width <= 0:
                continue
            if width > max_width:
                break
            if long_tmpl.ask <= 0:
                continue
            credit = short_tmpl.bid - long_tmpl.ask
            if credit < min_credit:
                continue
            pairs.append((short_tmpl, long_tmpl, width))
    return pairs


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
    for sp_tmpl, lp_tmpl, _ in _build_vertical_pairs(
        viable_puts, is_bull=True,
        delta_min=delta_min, delta_max=delta_max,
        max_width=max_width, min_credit=min_credit,
    ):
        short = replace(sp_tmpl, side="short", qty=1)
        long_ = replace(lp_tmpl, side="long",  qty=1)
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
    for sc_tmpl, lc_tmpl, _ in _build_vertical_pairs(
        viable_calls, is_bull=False,
        delta_min=delta_min, delta_max=delta_max,
        max_width=max_width, min_credit=min_credit,
    ):
        short = replace(sc_tmpl, side="short", qty=1)
        long_ = replace(lc_tmpl, side="long",  qty=1)
        idea = TradeIdea(
            symbol=symbol, expiry=expiry, dte=dte, strategy="bcs",
            legs=[short, long_], spot=spot, iv_ref=short.iv, hv30=hv30,
        )
        analyze(idea, r=r)
        ideas.append(idea)
    return ideas


# ── Four-leg: Iron Condor ────────────────────────────────────────────────────

def scan_iron_condors(viable_puts: list[Leg], viable_calls: list[Leg], *,
                      symbol: str, expiry: str, dte: int, spot: float,
                      hv30: float, delta_min: float, delta_max: float,
                      max_width: float = 20.0, min_credit: float = 0.05,
                      r: float = 0.05,
                      top_wings: int = 4) -> list[TradeIdea]:
    """Iron Condor = BPS (put wing) + BCS (call wing), same expiry.

    Neutral / range-bound: profit if spot stays between the short strikes.
    Defined risk on both wings (max loss = max(put wing loss, call wing loss)).

    Enumeration is pruned to O(top_wings²) by first ranking each wing by
    credit/width, then combining. Without pruning IC combos would be
    ~O(N²×M²) which explodes for liquid names.
    """
    ideas: list[TradeIdea] = []

    put_pairs = _build_vertical_pairs(
        viable_puts, is_bull=True,
        delta_min=delta_min, delta_max=delta_max,
        max_width=max_width, min_credit=min_credit,
    )
    call_pairs = _build_vertical_pairs(
        viable_calls, is_bull=False,
        delta_min=delta_min, delta_max=delta_max,
        max_width=max_width, min_credit=min_credit,
    )
    if not put_pairs or not call_pairs:
        return ideas

    # Rank each wing by credit-per-unit-of-risk; keep top-K.
    put_pairs.sort(key=lambda t: (t[0].bid - t[1].ask) / t[2], reverse=True)
    call_pairs.sort(key=lambda t: (t[0].bid - t[1].ask) / t[2], reverse=True)
    put_pairs  = put_pairs[:top_wings]
    call_pairs = call_pairs[:top_wings]

    for sp_tmpl, lp_tmpl, _pw in put_pairs:
        for sc_tmpl, lc_tmpl, _cw in call_pairs:
            # Sanity: short call strike must be above short put strike (otherwise
            # the "profitable range" is empty — you can't be between them).
            if sc_tmpl.strike <= sp_tmpl.strike:
                continue
            # Combined credit sanity check
            total = (sp_tmpl.bid - lp_tmpl.ask) + (sc_tmpl.bid - lc_tmpl.ask)
            if total < min_credit:
                continue
            sp = replace(sp_tmpl, side="short", qty=1)
            lp = replace(lp_tmpl, side="long",  qty=1)
            sc = replace(sc_tmpl, side="short", qty=1)
            lc = replace(lc_tmpl, side="long",  qty=1)
            idea = TradeIdea(
                symbol=symbol, expiry=expiry, dte=dte, strategy="ic",
                legs=[sp, lp, sc, lc],
                spot=spot,
                # Reference IV: average of the two short legs (they define
                # the profit range and dominate P calculations).
                iv_ref=(sp.iv + sc.iv) / 2.0,
                hv30=hv30,
            )
            analyze(idea, r=r)
            ideas.append(idea)
    return ideas


# ── Exotic: Jade Lizard ───────────────────────────────────────────────────────

def scan_jade_lizards(viable_puts: list[Leg], viable_calls: list[Leg], *,
                      symbol: str, expiry: str, dte: int, spot: float,
                      hv30: float, delta_min: float, delta_max: float,
                      max_width: float = 20.0, min_credit: float = 0.05,
                      r: float = 0.05,
                      top_wings: int = 6) -> list[TradeIdea]:
    """Jade Lizard = short put + short bear call spread.

    Magic feature: when total credit >= width of the call spread, the trade
    has NO upside risk. Profit zone covers the entire upside.
    """
    ideas: list[TradeIdea] = []
    short_puts = [p for p in viable_puts if delta_min <= abs(p.delta) <= delta_max]
    if not short_puts:
        return ideas

    bcs_pairs = _build_vertical_pairs(
        viable_calls, is_bull=False,
        delta_min=delta_min, delta_max=delta_max,
        max_width=max_width, min_credit=min_credit,
    )
    bcs_pairs.sort(key=lambda t: (t[0].bid - t[1].ask) / t[2], reverse=True)
    bcs_pairs = bcs_pairs[:top_wings]

    for sp_tmpl in short_puts:
        for sc_tmpl, lc_tmpl, _w in bcs_pairs:
            sp = replace(sp_tmpl, side="short", qty=1)
            sc = replace(sc_tmpl, side="short", qty=1)
            lc = replace(lc_tmpl, side="long",  qty=1)
            total = sp.bid + sc.bid - lc.ask
            if total < min_credit:
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
    out += scan_iron_condors(viable_puts, viable_calls,
                             symbol=symbol, expiry=expiry, dte=dte, spot=spot,
                             hv30=hv30, delta_min=delta_min, delta_max=delta_max,
                             max_width=max_width, min_credit=min_credit, r=r)
    out += scan_jade_lizards(viable_puts, viable_calls,
                             symbol=symbol, expiry=expiry, dte=dte, spot=spot,
                             hv30=hv30, delta_min=delta_min, delta_max=delta_max,
                             max_width=max_width, min_credit=min_credit, r=r)
    return out
