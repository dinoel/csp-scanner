"""
Skew scanner for US equity options via yfinance (delayed data).

Метрика: 25-дельта Risk Reversal = IV(put_25d) - IV(call_25d)  [в пунктах волы]
  > 0  : путы дороже коллов (нормальный страх-skew)
  < 0  : коллы дороже путов (инверсия — squeeze / M&A / жадность; топливо для COLLARS)
  rr_25d_pct = RR / ATM_IV * 100  — нормализация для cross-name сравнения

Usage:
    python skew_scanner.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

import skew as skewlib
from analytics import pick_expiry

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE_SIZE    = 600
DTE_MIN        = 25
DTE_MAX        = 45
RISK_FREE_RATE = 0.05
MIN_OI         = 100   # отсекаем неликвидные страйки при поиске дельт
CSV_OUT        = "skew_scan.csv"
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SkewRow:
    ticker:        str
    spot:          float
    expiry:        str
    dte:           int
    atm_iv:        float
    rr_25d:        float
    rr_25d_pct:    float
    rr_10d:        float
    put_25d_iv:    float
    call_25d_iv:   float
    put_25d_K:     float
    call_25d_K:    float
    put_25d_bid:   float
    put_25d_ask:   float
    put_25d_last:  float
    call_25d_bid:  float
    call_25d_ask:  float
    call_25d_last: float
    put_10d_iv:    float
    call_10d_iv:   float


def _filter_oi(df: pd.DataFrame, min_oi: int) -> pd.DataFrame:
    """Drop rows with open interest below threshold."""
    if "openInterest" not in df.columns:
        return df
    return df[df["openInterest"].fillna(0) >= min_oi].reset_index(drop=True)


def scan_ticker(symbol: str) -> Optional[SkewRow]:
    print(f"[{symbol}] fetching...")
    tk = yf.Ticker(symbol)

    try:
        spot = float(tk.history(period="5d", auto_adjust=True)["Close"].iloc[-1])
    except Exception:
        spot = 0.0
    if not spot or math.isnan(spot) or spot <= 0:
        print(f"[{symbol}] нет спота, пропуск")
        return None

    today  = datetime.now(timezone.utc).date()
    expiry, dte = pick_expiry(list(tk.options or []), today, DTE_MIN, DTE_MAX)
    if expiry is None:
        print(f"[{symbol}] нет экспирации в окне {DTE_MIN}-{DTE_MAX} DTE")
        return None

    chain      = tk.option_chain(expiry)
    calls      = _filter_oi(chain.calls, MIN_OI)
    puts       = _filter_oi(chain.puts,  MIN_OI)
    T          = dte / 365.0

    live = int((calls["bid"].fillna(0) > 0).sum())
    if live == 0:
        print(f"[{symbol}] нет живых котировок — данные от последних сделок, могут быть устаревшими")

    # Full details for the report
    _, p25_iv, p25_bid, p25_ask, p25_last = skewlib.find_at_delta(puts,  spot, T, RISK_FREE_RATE, 0.25, True)
    _, c25_iv, c25_bid, c25_ask, c25_last = skewlib.find_at_delta(calls, spot, T, RISK_FREE_RATE, 0.25, False)
    _, p10_iv, *_ = skewlib.find_at_delta(puts,  spot, T, RISK_FREE_RATE, 0.10, True)
    _, c10_iv, *_ = skewlib.find_at_delta(calls, spot, T, RISK_FREE_RATE, 0.10, False)
    p25_K, *_     = skewlib.find_at_delta(puts,  spot, T, RISK_FREE_RATE, 0.25, True)
    c25_K, *_     = skewlib.find_at_delta(calls, spot, T, RISK_FREE_RATE, 0.25, False)
    atm_iv        = skewlib.find_atm_iv(calls, spot, T, RISK_FREE_RATE)

    if math.isnan(atm_iv) or math.isnan(p25_iv) or math.isnan(c25_iv):
        print(f"[{symbol}] не хватает IV (atm={atm_iv:.3f}, p25={p25_iv}, c25={c25_iv})")
        return None

    rr25 = p25_iv - c25_iv
    rr10 = (p10_iv - c10_iv) if not (math.isnan(p10_iv) or math.isnan(c10_iv)) else float("nan")

    def _r2(v): return round(float(v), 2) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float("nan")

    return SkewRow(
        ticker=symbol, spot=round(spot, 2), expiry=expiry, dte=dte,
        atm_iv=round(atm_iv, 4),
        rr_25d=round(rr25, 4),
        rr_25d_pct=round(rr25 / atm_iv * 100.0, 2),
        rr_10d=round(rr10, 4) if not math.isnan(rr10) else float("nan"),
        put_25d_iv=round(p25_iv, 4),  call_25d_iv=round(c25_iv, 4),
        put_25d_K=p25_K or float("nan"), call_25d_K=c25_K or float("nan"),
        put_25d_bid=_r2(p25_bid),   put_25d_ask=_r2(p25_ask),  put_25d_last=_r2(p25_last),
        call_25d_bid=_r2(c25_bid), call_25d_ask=_r2(c25_ask), call_25d_last=_r2(c25_last),
        put_10d_iv=round(p10_iv, 4) if not math.isnan(p10_iv) else float("nan"),
        call_10d_iv=round(c10_iv, 4) if not math.isnan(c10_iv) else float("nan"),
    )


def get_sp500_tickers() -> list[str]:
    url = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
           "/refs/heads/main/data/constituents.csv")
    df = pd.read_csv(url)
    return [s.replace(".", "-") for s in df["Symbol"].tolist()]


def main():
    print("Загружаем список S&P 500...")
    all_tickers = get_sp500_tickers()
    tickers = random.sample(all_tickers, min(SAMPLE_SIZE, len(all_tickers)))
    print(f"Выбрано {len(tickers)} тикеров\n")

    rows = []
    for sym in tickers:
        try:
            r = scan_ticker(sym)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[{sym}] ошибка: {e!r}")

    if not rows:
        print("Пусто.")
        return

    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.sort_values("rr_25d", ascending=True).reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== SKEW SCAN ===")
    print(df.to_string(index=False))

    cols     = ["ticker", "spot", "dte", "atm_iv", "rr_25d", "rr_25d_pct", "rr_10d"]
    inverted = df[df.rr_25d < 0]
    steep    = df[df.rr_25d_pct > 15]
    flat     = df[(df.rr_25d_pct >= 0) & (df.rr_25d_pct < 3)]

    print("\n--- ИНВЕРСИЯ (calls > puts) ---")
    if len(inverted):
        print(inverted[cols].to_string(index=False))
        for _, r in inverted.iterrows():
            print(f"\n  {r.ticker}  spot={r.spot}  expiry={r.expiry}  dte={r.dte}")
            for label, K, iv, bid, ask, last in [
                ("PUT  25Δ", r.put_25d_K,  r.put_25d_iv,  r.put_25d_bid,  r.put_25d_ask,  r.put_25d_last),
                ("CALL 25Δ", r.call_25d_K, r.call_25d_iv, r.call_25d_bid, r.call_25d_ask, r.call_25d_last),
            ]:
                has_mkt   = not (math.isnan(bid) or math.isnan(ask))
                price_str = f"bid={bid:>6.2f}  ask={ask:>6.2f}" if has_mkt else f"last={last:>6.2f}  (no market)"
                print(f"    {label}  strike={K:>8.2f}  {price_str}  IV={iv*100:.1f}%")
    else:
        print("  нет имён")

    print("\n--- ПЛОСКИЙ skew (RR < 3% ATM) ---")
    print(flat[cols].to_string(index=False) if len(flat) else "  нет имён")

    print("\n--- КРУТОЙ PUT-SKEW (RR > 15% ATM) ---")
    print(steep[cols].to_string(index=False) if len(steep) else "  нет имён")

    df.to_csv(CSV_OUT, index=False)
    print(f"\nСохранено в {CSV_OUT}")


if __name__ == "__main__":
    main()
