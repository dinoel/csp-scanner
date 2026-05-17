"""
Skew scanner for US equity options via yfinance (delayed data).

Метрика: 25-дельта Risk Reversal = IV(put_25d) - IV(call_25d)  [в пунктах волы]
  > 0  : путы дороже коллов (нормальный страх-skew)
  < 0  : коллы дороже путов (инверсия — squeeze / M&A / жадность; топливо для COLLARS)
  rr_25d_pct = RR / ATM_IV * 100  — нормализация для cross-name сравнения

Дельта вычисляется через Black-Scholes из IV, которую возвращает Yahoo Finance.

Требования:
    pip install yfinance pandas
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

# -------------------- CONFIG --------------------
SAMPLE_SIZE = 600
DTE_MIN = 25
DTE_MAX = 45
RISK_FREE_RATE = 0.05   # приближение — можно обновить вручную
MIN_OPEN_INTEREST = 100  # отсекаем неликвидные страйки
CSV_OUT = "skew_scan.csv"
# ------------------------------------------------


@dataclass
class SkewRow:
    ticker: str
    spot: float
    expiry: str
    dte: int
    atm_iv: float
    rr_25d: float
    rr_25d_pct: float
    rr_10d: float
    put_25d_iv: float
    call_25d_iv: float
    put_25d_K: float
    call_25d_K: float
    put_25d_bid: float
    put_25d_ask: float
    put_25d_last: float
    call_25d_bid: float
    call_25d_ask: float
    call_25d_last: float
    put_10d_iv: float
    call_10d_iv: float


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_put: bool) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    nd1 = _norm_cdf(d1)
    return nd1 - 1.0 if is_put else nd1


def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_put: bool) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_put:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def compute_iv(S: float, K: float, T: float, r: float, price: float, is_put: bool) -> float:
    """BS IV via bisection from market price. Returns nan if price is outside arbitrage bounds."""
    intrinsic = max(0.0, (K - S if is_put else S - K) * math.exp(-r * T))
    if price <= intrinsic or price <= 0:
        return float("nan")
    lo, hi = 1e-4, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, r, mid, is_put) > price:
            hi = mid
        else:
            lo = mid
    iv = (lo + hi) / 2
    return iv if 0.01 < iv < 5.0 else float("nan")


def _market_price(row) -> tuple[float, bool]:
    """(price, is_live). is_live=True means bid/ask mid was used.
    Returns nan if open interest is below MIN_OPEN_INTEREST."""
    oi = row.get("openInterest") or 0
    if oi < MIN_OPEN_INTEREST:
        return float("nan"), False
    bid = row.get("bid") or float("nan")
    ask = row.get("ask") or float("nan")
    last = row.get("lastPrice") or float("nan")
    if not math.isnan(bid) and not math.isnan(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2, True
    if not math.isnan(last) and last > 0:
        return last, False
    return float("nan"), False


def pick_expiry(expirations, today, dte_min, dte_max):
    candidates = []
    for s in expirations:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            dte = (d - today).days
            if dte_min <= dte <= dte_max:
                candidates.append((dte, s))
        except ValueError:
            continue
    if not candidates:
        return None, None
    dte, s = min(candidates)
    return s, dte


def find_at_delta(df: pd.DataFrame, spot: float, T: float, r: float,
                  target_delta: float, is_put: bool):
    """Strike, IV, bid, ask, last of the option whose BS delta is closest to ±target_delta.
    IV is computed from market price (mid when live, last otherwise)."""
    best_strike = best_iv = best_bid = best_ask = best_last = None
    best_dist = float("inf")
    target_signed = -target_delta if is_put else target_delta
    for _, row in df.iterrows():
        price, _ = _market_price(row)
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
            best_strike = row["strike"]
            best_iv = iv
            best_bid = row.get("bid")
            best_ask = row.get("ask")
            best_last = row.get("lastPrice")
    return best_strike, best_iv, best_bid, best_ask, best_last


def find_atm_iv(calls: pd.DataFrame, spot: float, T: float, r: float) -> Optional[float]:
    best_iv = None
    best_dist = float("inf")
    for _, row in calls.iterrows():
        price, _ = _market_price(row)
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


def scan_ticker(symbol: str) -> Optional[SkewRow]:
    print(f"[{symbol}] fetching...")
    tk = yf.Ticker(symbol)

    spot = tk.fast_info.last_price
    if spot is None or math.isnan(spot) or spot <= 0:
        print(f"[{symbol}] нет спота, пропуск")
        return None

    today = datetime.now(timezone.utc).date()
    expiry, dte = pick_expiry(tk.options, today, DTE_MIN, DTE_MAX)
    if expiry is None:
        print(f"[{symbol}] нет экспирации в окне {DTE_MIN}-{DTE_MAX} DTE")
        return None

    chain = tk.option_chain(expiry)
    calls, puts = chain.calls, chain.puts

    T = dte / 365.0
    r = RISK_FREE_RATE

    live = sum(1 for _, row in calls.iterrows() if (row.get("bid") or 0) > 0)
    if live == 0:
        print(f"[{symbol}] нет живых котировок — данные от последних сделок, могут быть устаревшими")

    atm_iv = find_atm_iv(calls, spot, T, r)
    p25_K, p25_iv, p25_bid, p25_ask, p25_last = find_at_delta(puts,  spot, T, r, 0.25, is_put=True)
    c25_K, c25_iv, c25_bid, c25_ask, c25_last = find_at_delta(calls, spot, T, r, 0.25, is_put=False)
    p10_K, p10_iv, _, _, _ = find_at_delta(puts,  spot, T, r, 0.10, is_put=True)
    c10_K, c10_iv, _, _, _ = find_at_delta(calls, spot, T, r, 0.10, is_put=False)

    if not (atm_iv and p25_iv and c25_iv):
        print(f"[{symbol}] не хватает IV (atm={atm_iv}, p25={p25_iv}, c25={c25_iv})")
        return None

    rr25 = p25_iv - c25_iv
    rr10 = (p10_iv - c10_iv) if (p10_iv and c10_iv) else float("nan")

    return SkewRow(
        ticker=symbol,
        spot=round(spot, 2),
        expiry=expiry,
        dte=dte,
        atm_iv=round(atm_iv, 4),
        rr_25d=round(rr25, 4),
        rr_25d_pct=round(rr25 / atm_iv * 100.0, 2),
        rr_10d=round(rr10, 4) if not math.isnan(rr10) else float("nan"),
        put_25d_iv=round(p25_iv, 4),
        call_25d_iv=round(c25_iv, 4),
        put_25d_K=p25_K,
        call_25d_K=c25_K,
        put_25d_bid=round(p25_bid, 2) if p25_bid else float("nan"),
        put_25d_ask=round(p25_ask, 2) if p25_ask else float("nan"),
        put_25d_last=round(p25_last, 2) if p25_last else float("nan"),
        call_25d_bid=round(c25_bid, 2) if c25_bid else float("nan"),
        call_25d_ask=round(c25_ask, 2) if c25_ask else float("nan"),
        call_25d_last=round(c25_last, 2) if c25_last else float("nan"),
        put_10d_iv=round(p10_iv, 4) if p10_iv else float("nan"),
        call_10d_iv=round(c10_iv, 4) if c10_iv else float("nan"),
    )


def get_sp500_tickers() -> list[str]:
    #return ["TTWO", "NDAQ", "KHC"]
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/refs/heads/main/data/constituents.csv"
    df = pd.read_csv(url)
    return [s.replace(".", "-") for s in df["Symbol"].tolist()]


def main():
    print("Загружаем список S&P 500...")
    all_tickers = get_sp500_tickers()
    tickers = random.sample(all_tickers, min(SAMPLE_SIZE, len(all_tickers)))
    print(f"Выбрано {len(tickers)} тикеров: {', '.join(tickers)}\n")

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

    inverted = df[df.rr_25d < 0]
    steep    = df[df.rr_25d_pct > 15]
    flat     = df[(df.rr_25d_pct >= 0) & (df.rr_25d_pct < 3)]

    cols = ["ticker", "spot", "dte", "atm_iv", "rr_25d", "rr_25d_pct", "rr_10d"]

    print("\n--- ИНВЕРСИЯ (calls > puts) ---")
    if len(inverted):
        print(inverted[cols].to_string(index=False))
        print()
        for _, r in inverted.iterrows():
            print(f"  {r.ticker}  spot={r.spot}  expiry={r.expiry}  dte={r.dte}")
            for label, K, iv, bid, ask, last in [
                ("PUT  25Δ", r.put_25d_K,  r.put_25d_iv,  r.put_25d_bid,  r.put_25d_ask,  r.put_25d_last),
                ("CALL 25Δ", r.call_25d_K, r.call_25d_iv, r.call_25d_bid, r.call_25d_ask, r.call_25d_last),
            ]:
                has_market = not (math.isnan(bid) or math.isnan(ask))
                price_str = f"bid={bid:>6.2f}  ask={ask:>6.2f}" if has_market else f"last={last:>6.2f}  (no market)"
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
