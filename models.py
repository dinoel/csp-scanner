"""Shared data models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    fundamental_score: float  # 0-100 (NaN if FINANCIALDATASETS_API_KEY not set)
    rev_growth:    float      # YoY revenue growth %
    eps_beat_rate: float      # % of last 8 quarters beating estimates
    fcf_margin:    float      # free cash flow / revenue %
    bull_pct:      float      # StockTwits % bullish of tagged messages (NaN if no data)
    rsi:           float      # 14-day Wilder RSI (0-100)
    pcr:           float      # put/call volume ratio for selected expiry
    rr_25d_pct:    float      # 25-delta risk reversal normalised (%) — put skew
    score: float          # composite score (higher = better)
    company_name: str = ""  # longName from yfinance; "" if unavailable

    # ── Bull Put Spread extras (None when single-leg CSP) ──────────────────
    # When present, this row represents a vertical bull put spread:
    #   sell put @ `strike`     (the short leg — uses bid/ask/delta/iv as-is)
    #   buy  put @ `long_strike` (the long leg, OTM further)
    # `ret` / `ann_rtn` / `profit_prob` are recomputed on margin (max_loss),
    # so the existing display columns work without changes.
    long_strike: Optional[float] = None
    long_bid:    Optional[float] = None
    long_ask:    Optional[float] = None
    long_delta:  Optional[float] = None
    long_iv:     Optional[float] = None
    width:       Optional[float] = None   # short_strike - long_strike
    credit:      Optional[float] = None   # short_bid - long_ask, per share
    max_loss:    Optional[float] = None   # (width - credit) * 100, per contract
