# Options Scanner

Cash-Secured Put (CSP) scanner for US equities with rich HTML reports,
optional fundamentals analysis (via [financialdatasets.ai](https://financialdatasets.ai)),
StockTwits sentiment, and Claude-powered AI commentary on top candidates.

Two entry points:

- **`csp_scanner.py`** — the main short-put scanner (scoring, HTML report, AI analysis)
- **`skew_scanner.py`** — standalone 25-delta Risk Reversal scanner

Default data source is Yahoo Finance (15–20 min delayed, free).

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone <repo-url> csp-scanner
cd csp-scanner

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install yfinance pandas numpy requests anthropic
```

`anthropic` is only needed if `ENABLE_AI_ANALYSIS = True`.

### Optional env vars

| Variable | What it unlocks |
|---|---|
| `ANTHROPIC_API_KEY` | Claude AI analysis of top N candidates |
| `FINANCIALDATASETS_API_KEY` | Fundamentals (EPS history, revenue growth, FCF, balance sheet, news) |
| `STOCKTWITS_TOKEN` | StockTwits bullish-sentiment % (their public API now requires a token) |
| `MASSIVE_API_KEY` | Real-time quotes via `DATA_PROVIDER = "massive"` (paid plan) |

Everything works without these — the scanner gracefully degrades.

---

## Usage

### Short Put Scanner

```bash
python csp_scanner.py
```

Generates `results/csp_scan_<profile>.html` (one file per profile in `PROFILES_TO_RUN`)
and opens automatically in the browser. The CSV `csp_scan.csv` is also written.

### Pre-screener

```bash
python screener.py           # show liquid US equity candidates
python screener.py --profile low
```

Used internally by `csp_scanner.py` when `UNIVERSE = "screener"`. Filters by
price range, average volume, and beta via Yahoo's screener API. **MA200 filter
is intentionally not applied here** — it's handled asymmetrically per candidate
inside the scanner's composite score (`ma200_score`), so stocks slightly below
their 200-day MA still have a chance to surface.

### Skew Scanner

```bash
python skew_scanner.py
```

Standalone scanner for ranking 25-delta Risk Reversal — fear/greed in options.
Shares the underlying math with `csp_scanner.py` (`skew.py`).

### Tests

```bash
pytest tests/
```

---

## Configuration (`csp_scanner.py`)

Top of file:

```python
UNIVERSE: str | list[str] = "screener"   # "screener" | "sp500" | "russell2000" | ["AAPL", ...]
SAMPLE_SIZE: int | None   = None         # random sample size; None = full universe
RISK_PROFILE              = "medium"     # "low" | "medium" | "high"
PROFILES_TO_RUN           = ["medium"]   # list — generate one HTML per profile
ENABLE_AI_ANALYSIS        = True
AI_TOP_N                  = 10           # top candidates sent to Claude + fundamentals
AI_BATCH_SIZE             = 10           # hard cap per batch; prompts before next batch
AI_MODEL                  = "claude-opus-4-7"
DATA_PROVIDER             = "yfinance"   # "yfinance" | "massive"
ENABLE_FUNDAMENTALS       = True         # requires FINANCIALDATASETS_API_KEY
MARKET_OPEN_GRACE_MIN     = 30           # grace period after market open for volume freshness
```

### Risk Profiles

| | `low` | `medium` | `high` |
|---|---|---|---|
| DTE range | 21–51 | 7–51 | 7–31 |
| Delta range | 0.05–0.20 | 0.05–0.50 | 0.15–0.50 |
| Max strike | $150 | $200 | $200 |
| Min open interest | 100 | 50 | 50 |
| Min volume | 25 | 200 | 200 |
| Max spread | 30% | 50% | 30% |

- **low** — far OTM, long DTE, conservative.
- **medium** — balanced default.
- **high** — closer to ATM, short DTE, maximizes annualized return.

---

## What gets analyzed for each candidate

For every `(ticker, expiry, strike)` triple that survives liquidity filters:

- **Greeks & pricing**: Black-Scholes delta, theta, IV (bisection), profit prob
- **Liquidity**: open interest, today's volume (with smart freshness check), bid/ask spread
- **Trend**: % above/below 200-day MA → `ma200_score`
- **Volatility context**: IV Rank (52-week), HV30 (30-day realized), Expected Move
- **Skew**: 25-delta Risk Reversal vs ATM IV → put skew gauge
- **Analyst**: mean recommendation, target price, % upside (yfinance)
- **Earnings**: flag if earnings fall inside the expiry window

### Top N enrichment (only for ranked top `AI_TOP_N`)

To keep API spend bounded:

- **StockTwits** bullish sentiment %
- **financialdatasets.ai** — annual income statement, cash flow, balance sheet,
  last 8 quarters of EPS actual vs estimate, recent news headlines
- **Fundamental score** 0–100 = 35% EPS beat rate + 25% revenue growth +
  25% FCF margin + 15% debt/equity
- **RSI(14)** — Wilder's, from price history
- **Put/Call ratio** — from option chain volumes for the selected expiry
- **Claude analysis** — one paragraph + Risk/Verdict per candidate, plus market observation

After each batch of `AI_BATCH_SIZE` you're prompted to continue (so a bumped
`AI_TOP_N` won't silently burn tokens).

---

## Composite Score

```
score = SCORE_W_RETURN      × min(AnnRtn × PProb/100, 200)   # expected ann. return
      + SCORE_W_PROB        × PProb                            # probability of profit
      + SCORE_W_SAFETY      × min(vsEM_buffer, 150)            # safety margin vs expected move
      + SCORE_W_MA200       × ma200_score                      # trend strength (asymmetric)
      + SCORE_W_FUNDAMENTAL × fundamental_score                # only for top N
```

Weights are per-profile in `_PROFILES` inside `csp_scanner.py`.

---

## HTML Report

`results/csp_scan_<profile>.html` is fully self-contained (no external assets except finviz chart images).

Features:

- **Per-table column visibility**: above every table, a grid of checkboxes lets
  you toggle which columns to show. Each label has a tooltip explaining the
  metric. Selection is saved to `localStorage` per table.
- **All tables share the same 36-column set** — IV-focused tables hide
  fundamentals/sentiment by default but you can switch them on.
- **Sortable** — click any header. Sort survives column hide/show.
- **Hover ticker** → live finviz chart popup (smart positioning, avoids viewport edges).
- **Row background** shaded by composite score (darker → brighter green).
- **AI analysis block** at the top with Claude's commentary.
- **Sections**: Earnings-window candidates, All Results, HIGH IVR + IV>HV30, HIGH IVR Only, IV>HV30 Only.

---

## Data Providers

Set `DATA_PROVIDER` in config; everything else stays the same.

| Provider | Notes |
|---|---|
| `yfinance` | Free, 15–20 min delayed. Default. |
| `massive` | Real-time (paid). Requires `MASSIVE_API_KEY` (Polygon.io account). |

The interface lives in `providers/base.py`; add new providers by subclassing
`DataProvider` and registering in `providers/__init__.py`.

---

## Caching

Per-key SQLite cache (`yf_cache_v2.db`) keyed by data type with per-prefix TTLs:

| Prefix | What | TTL |
|---|---|---|
| `fast_info`, `options_list` | Spot + expiry list | 2 h |
| `chain` | Option chain per expiry | 2 h |
| `history_1y` | Price history | 24 h |
| `calendar` | Earnings dates | 24 h |
| `sp500`, `r2k` | Index constituents | 24 h |
| `fd_income`, `fd_cashflow`, `fd_balance` | Fundamentals | 3 d |
| `fd_earnings` | Quarterly EPS history | 24 h |
| `fd_news` | News headlines | 1 h |
| `sentiment` | StockTwits | 1 h |

Delete `yf_cache_v2.db` to force a full refresh.

---

## Project Layout

```
csp_scanner.py        # main orchestrator
skew_scanner.py       # standalone skew scanner

analytics.py          # pure math: BS, MA200, HV30, IV rank, RSI, expected move, expiry pick
models.py             # PutRow dataclass
report.py             # HTML generation + column toggles + finviz hover
ai.py                 # Claude prompt construction + analysis
fundamentals.py       # financialdatasets.ai client + 0-100 fundamental score
sentiment.py          # StockTwits client
skew.py               # 25-delta Risk Reversal math (shared)
cache.py              # SQLite cache with per-key TTLs
screener.py           # Yahoo EquityQuery pre-filter

providers/            # DataProvider abstraction
  base.py             # ABC
  yfinance_provider.py
  massive_provider.py

tests/                # pytest test suite
```

---

## Notes

- Data is **15–20 minutes delayed** by default (Yahoo Finance free tier).
- Option `volume` from yfinance reflects the **last session the contract traded**,
  not today. The scanner uses `lastTradeDate` + `MARKET_OPEN_GRACE_MIN` to
  reject stale contracts (in weekends and pre-open, previous business day is OK).
- If you see `rate limited` errors from Yahoo, reduce `MAX_WORKERS` to 3–4.
- `[!]` next to a symbol = earnings within the expiry window — higher premium,
  binary event risk.
