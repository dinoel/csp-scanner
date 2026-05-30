"""Shared data models."""
from __future__ import annotations

from dataclasses import dataclass


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
