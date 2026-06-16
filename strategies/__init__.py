"""Trade-idea generator: scan many strategies per ticker, score uniformly.

The infrastructure piece consists of:
  - `Leg` + `TradeIdea` dataclasses (generic, support any leg count/type)
  - `numerical_ev()` — integrate payoff × lognormal pdf
  - `analyze()` — compute all metrics (credit, max gain/loss, breakevens, EV
    variants, ROI_ann) given just a list of legs + spot/T/r/IV/HV30

The scanner piece lives in `strategies.scanners`; each scanner enumerates
viable strike combinations and returns `list[TradeIdea]`.
"""
from .base import Leg, TradeIdea, analyze, numerical_ev, find_candidate_legs
from .scanners import scan_csps, scan_bps, scan_jade_lizards, scan_all

__all__ = [
    "Leg", "TradeIdea",
    "analyze", "numerical_ev", "find_candidate_legs",
    "scan_csps", "scan_bps", "scan_jade_lizards", "scan_all",
]
