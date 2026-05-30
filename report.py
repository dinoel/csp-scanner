"""HTML report generation for the CSP scanner.

Emits a "trader-grade" React-based UI. The HTML shell + styles + React app
live in `report_assets/` and are mirrored once into `results/_assets/`; each
scan folder only contains `data.js` (all three profiles in one file).
"""
from __future__ import annotations

import datetime
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from ai import _md_to_html

RATING_ABBREV = {
    "strong_buy": "STR_BUY", "buy": "BUY", "hold": "HOLD",
    "underperform": "UNDP",   "sell": "SELL",
}
RATING_CLASS = {
    "strong_buy": "r-sb", "buy": "r-b", "hold": "r-h",
    "underperform": "r-s", "sell": "r-s",
}

# CSS used by csp_scanner._build_index_html for the top-level results index page.
# The scanner report itself uses report_assets/scanner.css.
_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font:12px/1.5 'Consolas','Courier New',monospace;padding:20px}
h1{color:#58a6ff;font-size:16px;margin-bottom:4px}
h2{color:#3fb950;font-size:12px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid #21262d;text-transform:uppercase;letter-spacing:.05em}
.meta{color:#8b949e;font-size:11px;margin-bottom:20px}
.wrap{overflow-x:auto;margin-bottom:4px}
table{border-collapse:collapse;white-space:nowrap}
thead th{background:#161b22;color:#58a6ff;padding:5px 10px;border-bottom:2px solid #30363d;text-align:right;position:sticky;top:0;z-index:1}
thead th:first-child{text-align:left}
tbody td{padding:3px 10px;text-align:right;border-bottom:1px solid #161b22;color:#e6edf3}
tbody td:first-child{text-align:left}
tbody tr:hover td{filter:brightness(1.5)}
.nav{margin-bottom:14px;padding:6px 0;border-bottom:1px solid #21262d;font-size:11px}
.nav a{color:#58a6ff;text-decoration:none;margin:0 4px}
.nav a:hover{text-decoration:underline}
.nav .sep{color:#30363d;margin:0 2px}
.nav .cur{color:#e6edf3;font-weight:700;margin:0 4px}
.idx-table{margin-top:8px}
.idx-table td a{color:#58a6ff;text-decoration:none}
.idx-table td a:hover{text-decoration:underline}
"""

_ASSETS_DIR = Path(__file__).parent / "report_assets"
_SHARED_ASSET_NAMES = ("scan.html", "scanner.css", "scanner.jsx")


def sync_assets(results_root: Path | str = Path("results")) -> None:
    """Mirror report_assets/{scan.html,scanner.css,scanner.jsx} into <results_root>/_assets/.

    Copies a file only if missing or its bytes differ — running it repeatedly
    is cheap. Exposed as a CLI (`python -m report sync [dir]`) so the user can
    refresh shared assets after editing report_assets/ without rerunning a scan.
    """
    root = Path(results_root)
    assets_dir = root / "_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for name in _SHARED_ASSET_NAMES:
        src = _ASSETS_DIR / name
        dst = assets_dir / name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dst)
            print(f"Synced {dst}")


def _safe_float(v):
    """Convert pandas/numpy NaN and non-finite floats to None for JSON."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return v


def _row_to_js(rec: dict) -> dict:
    """Map a DataFrame row record to the JS row shape expected by scanner.jsx."""
    a_rating_raw = rec.get("analyst_rating")
    rating = RATING_ABBREV.get(str(a_rating_raw), str(a_rating_raw) if a_rating_raw else "none")
    rating_n = int(rec.get("analyst_num") or 0) or None
    return {
        "sym":       rec.get("symbol"),
        "name":      rec.get("company_name") or "",
        "earn":      bool(rec.get("earnings_date") or ""),
        "score":     _safe_float(rec.get("score")),
        "price":     _safe_float(rec.get("price")),
        "rating":    rating,
        "ratingN":   rating_n,
        "target":    _safe_float(rec.get("analyst_target")),
        "upsidePct": _safe_float(rec.get("analyst_upside")),
        "fundScore": _safe_float(rec.get("fundamental_score")),
        "revGrow":   _safe_float(rec.get("rev_growth")),
        "beatRate":  _safe_float(rec.get("eps_beat_rate")),
        "fcf":       _safe_float(rec.get("fcf_margin")),
        "stBull":    _safe_float(rec.get("bull_pct")),
        "rsi":       _safe_float(rec.get("rsi")),
        "pcr":       _safe_float(rec.get("pcr")),
        "skew":      _safe_float(rec.get("rr_25d_pct")),
        "exp":       rec.get("exp_date") or None,
        "dte":       _safe_float(rec.get("dte")),
        "strike":    _safe_float(rec.get("strike")),
        "mnessPct":  _safe_float(rec.get("moneyness")),
        "emPct":     _safe_float(rec.get("exp_move_pct")),
        "vsEm":      _safe_float(rec.get("vs_em")),
        "bid":       _safe_float(rec.get("bid")),
        "ask":       _safe_float(rec.get("ask")),
        "spread":    _safe_float(rec.get("spread")),
        "vol":       _safe_float(rec.get("volume")),
        "beBid":     _safe_float(rec.get("be_bid")),
        "bePct":     _safe_float(rec.get("pct_be_bid")),
        "oi":        _safe_float(rec.get("open_int")),
        "ivr":       _safe_float(rec.get("iv_rank")),
        "iv":        _safe_float(rec.get("iv")),
        "hv30":      _safe_float(rec.get("hv30")),
        "delta":     _safe_float(rec.get("delta")),
        "theta":     _safe_float(rec.get("theta")),
        "retPct":    _safe_float(rec.get("ret")),
        "annRtn":    _safe_float(rec.get("ann_rtn")),
        "pProb":     _safe_float(rec.get("pProb")) if "pProb" in rec else _safe_float(rec.get("profit_prob")),
        "ma200":     _safe_float(rec.get("ma200_pct")),
    }


def _build_scan_meta(df: pd.DataFrame, *, profile: str | None, generated: str,
                     universe: str | None, dte_min, dte_max,
                     delta_min, delta_max, strike_max, vol_min, spread_max_pct,
                     index_href: str | None,
                     failed: list[str]) -> dict:
    return {
        "universe":     universe or "—",
        "profile":      profile or "medium",
        "dteMin":       dte_min, "dteMax": dte_max,
        "deltaMin":     delta_min, "deltaMax": delta_max,
        "strikeMax":    strike_max,
        "volMin":       vol_min,
        "spreadMaxPct": spread_max_pct,
        "generated":    generated,
        "indexHref":    index_href,
        "failed":       list(failed or []),
        "version":      "v." + datetime.datetime.now().strftime("%Y.%m"),
    }


def _build_scan_tabs(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"earnings": [], "all": [], "premium": [], "highIvr": [], "ivOverHv": []}
    earn      = df["earnings_date"].ne("")
    no_earn   = ~earn
    high_ivr  = df["iv_rank"] >= 50
    iv_gt_hv  = df["iv"] > df["hv30"]
    return {
        "earnings": df.loc[earn, "symbol"].tolist(),
        "all":      df["symbol"].tolist(),
        "premium":  df.loc[high_ivr &  iv_gt_hv & no_earn, "symbol"].tolist(),
        "highIvr":  df.loc[high_ivr & ~iv_gt_hv & no_earn, "symbol"].tolist(),
        "ivOverHv": df.loc[~high_ivr &  iv_gt_hv & no_earn, "symbol"].tolist(),
    }


def _parse_meta_from_config(config_str: str) -> dict:
    """Best-effort parse of the legacy config_str into structured fields.

    Format produced by csp_scanner.py:
      Universe: X  |  Profile: P  |  DTE: A-B  |  |Δ|: C-D  |
      strike≤E  |  vol≥F  |  spread≤G%
    """
    out = {"universe": None, "profile": None,
           "dte_min": None, "dte_max": None,
           "delta_min": None, "delta_max": None,
           "strike_max": None, "vol_min": None, "spread_max_pct": None}
    for part in [p.strip() for p in config_str.split("|") if p.strip()]:
        low = part.lower()
        try:
            if low.startswith("universe:"):
                out["universe"] = part.split(":", 1)[1].strip()
            elif low.startswith("profile:"):
                out["profile"] = part.split(":", 1)[1].strip()
            elif low.startswith("dte:"):
                a, b = part.split(":", 1)[1].split("-")
                out["dte_min"] = float(a); out["dte_max"] = float(b)
            elif "Δ" in part:
                a, b = part.split(":", 1)[1].split("-")
                out["delta_min"] = float(a); out["delta_max"] = float(b)
            elif "strike" in low:
                out["strike_max"] = float(part.split("≤")[1])
            elif "vol" in low:
                out["vol_min"] = float(part.split("≥")[1])
            elif "spread" in low:
                v = part.split("≤")[1].strip().rstrip("%")
                out["spread_max_pct"] = float(v)
        except (ValueError, IndexError):
            continue
    return out


def build_profile_block(df: pd.DataFrame, config_str: str, failed: list,
                        ai_text: str | None = None, ai_top_n: int = 10,
                        profile: str | None = None,
                        index_href: str | None = None) -> dict:
    """Build the JS-shaped data block for one profile.

    Returns `{"meta": {...}, "rows": [...], "tabs": {...}, "ai": {...}|None}`.
    Pure: does not touch the filesystem. Caller accumulates blocks across
    profiles and hands them to `write_scan_bundle`.
    """
    parsed = _parse_meta_from_config(config_str)
    profile = profile or parsed["profile"] or "medium"
    universe = parsed["universe"]
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return {
        "meta": _build_scan_meta(
            df,
            profile=profile,
            generated=generated,
            universe=universe,
            dte_min=parsed["dte_min"], dte_max=parsed["dte_max"],
            delta_min=parsed["delta_min"], delta_max=parsed["delta_max"],
            strike_max=parsed["strike_max"], vol_min=parsed["vol_min"],
            spread_max_pct=parsed["spread_max_pct"],
            index_href=index_href,
            failed=failed,
        ),
        "rows": [_row_to_js(rec) for rec in df.to_dict("records")],
        "tabs": _build_scan_tabs(df),
        "ai":   {"html": _md_to_html(ai_text), "topN": ai_top_n} if ai_text else None,
    }


def write_scan_bundle(scan_dir: Path | str, scan_data: dict) -> None:
    """Write <scan_dir>/data.js with `window.SCAN_DATA = {profile: block, ...}`
    and refresh shared assets under <scan_dir>/../_assets/."""
    scan_dir = Path(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)
    data_js = "window.SCAN_DATA = " + json.dumps(scan_data) + ";\n"
    (scan_dir / "data.js").write_text(data_js, encoding="utf-8")
    sync_assets(scan_dir.parent)
    print(f"Saved data to {scan_dir}/data.js  ({len(scan_data)} profiles)")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "sync":
        root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
        sync_assets(root)
    else:
        print("Usage: python -m report sync [results_dir]")
        sys.exit(1)
