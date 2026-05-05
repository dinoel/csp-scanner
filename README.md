# Options Scanner

Two scanners for US equity options using free delayed data from Yahoo Finance.

- **`csp_scanner.py`** — Short Put (Cash-Secured Put) scanner

No API keys required.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/dinoel/csp-scanner.git
cd csp-scanner

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install yfinance pandas numpy
```

---

## Usage

### Short Put Scanner

```bash
python csp_scanner.py
```

Scans the universe, scores every (expiry × strike) combination, and prints the top results sorted by composite score. Results are also saved to `csp_scan.csv`.

---

## Configuration (`csp_scanner.py`)

All settings are at the top of the file:

```python
UNIVERSE     = "sp500"    # "sp500" | "russell2000" | "sp500+russell2000" | ["AAPL", "NVDA", ...]
SAMPLE_SIZE  = 501        # random sample size; None = full universe
RISK_PROFILE = "medium"   # "low" | "medium" | "high"
COMPUTE_IV_RANK = True    # False = faster, skips 1yr history download
MAX_WORKERS  = 6          # parallel workers; reduce if you hit Yahoo rate limits
```

### Risk Profiles

| | `low` | `medium` | `high` |
|---|---|---|---|
| DTE range | 21–51 | 7–51 | 7–31 |
| Delta range | 0.05–0.20 | 0.05–0.50 | 0.15–0.50 |
| Max strike | $150 | $200 | $200 |
| Min volume | 25 | 10 | 10 |
| Max spread | 30% | 50% | 60% |
| Score: prob weight | 0.55 | 0.45 | 0.30 |
| Score: return weight | 0.15 | 0.25 | 0.45 |

- **low** — far OTM, long DTE, strict liquidity filters. Probability of profit 85–95%.
- **medium** — balanced default.
- **high** — closer to ATM, short DTE, maximizes annualized return.

---

## Output Columns

| Column | Description |
|---|---|
| Symbol | Ticker (+ `[!]` if earnings within expiry window) |
| Score | Composite ranking score (higher = better) |
| Price~ | Current stock price (15–20 min delayed) |
| Exp Date | Option expiration date |
| DTE | Days to expiration |
| Strike | Put strike price |
| Mness% | Moneyness: `(strike − price) / price × 100` |
| EM% | Expected move: ATM straddle price as % of spot |
| vs EM% | Strike distance as % of expected move (>100 = outside EM, safer) |
| Bid / Ask / Spread | Option quotes |
| BE(Bid) | Break-even price = `strike − bid` |
| %BE | Downside cushion: `(price − BE) / price × 100` |
| Vol / OI | Volume and open interest |
| IVR | IV Rank: where current IV sits in 52-week range (0–100) |
| IV% | Implied volatility at the strike |
| Delta | Put delta (negative) |
| θ/day | Daily theta — $ earned per share per day from time decay |
| Ret% | Return on capital: `bid / strike × 100` |
| Ann Rtn% | Annualized return |
| PProb% | Probability of profit: N(d₂) from Black-Scholes |
| MA200% | Distance from 200-day MA (positive = above, bullish) |
| Earn | Earnings date if it falls within the expiry window |

---

## Scoring

Each `(expiry, strike)` pair is scored with:

```
Score = 0.25 × min(AnnRtn × PProb/100, 200)   # expected return
      + 0.45 × PProb                            # probability of profit
      + 0.20 × min(vsEM, 150)                   # safety buffer vs expected move
      + 0.10 × MA200score                       # trend (0–100)
```

Weights are defined per profile and can be customized in `_PROFILES`.

---

## Caching

Results are cached in `yf_cache_v2.db` (SQLite + pickle) to avoid redundant API calls:

| Data type | TTL |
|---|---|
| Price / options list | 2 hours |
| Options chain | 2 hours |
| 1yr price history | 24 hours |
| Earnings calendar | 24 hours |
| S&P 500 / Russell 2000 list | 24 hours |

Delete `yf_cache_v2.db` to force a full refresh.

---

## Notes

- Data is **15–20 minutes delayed** (Yahoo Finance free tier).
- Run during market hours for live bid/ask quotes. After hours, the scanner falls back to last trade prices with stricter volume filters.
- If you see `rate limited` errors, reduce `MAX_WORKERS` to 3–4.
- `[!]` next to a symbol means earnings are announced before option expiry — higher premium but binary event risk.
