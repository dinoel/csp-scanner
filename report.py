"""HTML report generation for the CSP scanner."""
from __future__ import annotations

import datetime
import html as _he
import math

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

_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font:12px/1.5 'Consolas','Courier New',monospace;padding:20px}
h1{color:#58a6ff;font-size:16px;margin-bottom:4px}
h2{color:#3fb950;font-size:12px;margin:28px 0 8px;padding-bottom:4px;border-bottom:1px solid #21262d;text-transform:uppercase;letter-spacing:.05em}
.meta{color:#8b949e;font-size:11px;margin-bottom:20px}
.wrap{overflow-x:auto;margin-bottom:4px}
table{border-collapse:collapse;white-space:nowrap}
thead th{background:#161b22;color:#58a6ff;padding:5px 10px;cursor:pointer;user-select:none;border-bottom:2px solid #30363d;text-align:right;position:sticky;top:0;z-index:1}
thead th:first-child{text-align:left}
thead th:hover{background:#1f2937}
thead th.asc::after{content:' ▲';font-size:9px;opacity:.8}
thead th.desc::after{content:' ▼';font-size:9px;opacity:.8}
tbody td{padding:3px 10px;text-align:right;border-bottom:1px solid #161b22;color:#e6edf3}
tbody td:first-child{text-align:left}
tbody tr:hover td{filter:brightness(1.5)}
.earn{color:#f0883e!important;font-weight:700}
.r-sb{color:#56d364;font-weight:700}.r-b{color:#3fb950}
.r-h{color:#d29922}.r-s{color:#f85149}
.empty{color:#6e7681;font-style:italic;font-size:11px;padding:4px 0}
.failed{color:#f85149;font-size:11px;margin-top:24px}
.ai-box{background:#111827;border:1px solid #30363d;border-left:3px solid #58a6ff;
        border-radius:4px;padding:14px 18px;margin:8px 0 4px;max-width:900px;
        font-size:12px;line-height:1.7;color:#c9d1d9}
.ai-box h3{color:#58a6ff;font-size:12px;margin:10px 0 4px}
.ai-box strong{color:#e6edf3}
.ai-box p{margin:0 0 6px}
.s-bull{color:#56d364;font-weight:700}.s-bear{color:#f85149;font-weight:700}.s-neu{color:#d29922}
.rsi-ob{color:#f0883e;font-weight:700}.rsi-os{color:#f85149;font-weight:700}
.pcr-hi{color:#f0883e}.pcr-lo{color:#56d364}
.skew-steep{color:#f0883e}.skew-inv{color:#56d364;font-weight:700}
.fv{display:inline-block}
.fv a{color:inherit;text-decoration:none}
.fv a:hover{text-decoration:underline}
.nav{margin-bottom:14px;padding:6px 0;border-bottom:1px solid #21262d;font-size:11px}
.nav a{color:#58a6ff;text-decoration:none;margin:0 4px}
.nav a:hover{text-decoration:underline}
.nav .sep{color:#30363d;margin:0 2px}
.nav .cur{color:#e6edf3;font-weight:700;margin:0 4px}
.idx-table{margin-top:8px}
.idx-table td a{color:#58a6ff;text-decoration:none}
.idx-table td a:hover{text-decoration:underline}
#fv-pop{display:none;position:fixed;z-index:9999;
        background:#161b22;border:1px solid #30363d;border-radius:6px;
        padding:6px;box-shadow:0 12px 32px #000a;pointer-events:none}
#fv-pop img{display:block;width:440px;height:255px;border-radius:3px}
.col-toggles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
             gap:2px 12px;margin:4px 0 6px;padding:8px 10px;background:#0f1419;
             border:1px solid #21262d;border-radius:4px;font-size:11px}
.col-toggles label{display:flex;align-items:center;gap:6px;cursor:help;color:#c9d1d9;
                   padding:2px 4px;border-radius:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.col-toggles label:hover{background:#161b22;color:#58a6ff}
.col-toggles input{cursor:pointer;margin:0;accent-color:#58a6ff;flex-shrink:0}
"""

_JS = """\
document.querySelectorAll('table[id]').forEach(t=>{
  const hs=[...t.querySelectorAll('thead th')];let s={c:-1,a:1};
  hs.forEach((h,i)=>{
    h.addEventListener('click',()=>{
      const a=s.c===i?-s.a:1;s={c:i,a};
      hs.forEach(x=>x.classList.remove('asc','desc'));
      h.classList.add(a>0?'asc':'desc');
      const tb=t.querySelector('tbody');
      [...tb.rows].sort((x,y)=>{
        const av=x.cells[i].dataset.v,bv=y.cells[i].dataset.v;
        const an=parseFloat(av),bn=parseFloat(bv);
        const d=(!isNaN(an)&&!isNaN(bn))?an-bn:String(av).localeCompare(String(bv));
        return a*d;
      }).forEach(r=>tb.append(r));
    });
  });
});
document.querySelectorAll('.col-toggles').forEach(tb=>{
  const tid=tb.dataset.target;
  const tbl=document.getElementById(tid);
  if(!tbl)return;
  const key='cols:'+tid;
  let saved={};
  try{saved=JSON.parse(localStorage.getItem(key)||'{}');}catch(e){}
  const apply=(col,show)=>{
    tbl.querySelectorAll('[data-col="'+col+'"]').forEach(c=>{
      c.style.display=show?'':'none';
    });
  };
  tb.querySelectorAll('input').forEach(cb=>{
    const col=cb.dataset.col;
    if(col in saved){
      const show=!!saved[col];
      cb.checked=show;
      apply(col,show);
    }
    cb.addEventListener('change',()=>{
      apply(col,cb.checked);
      saved[col]=cb.checked?1:0;
      localStorage.setItem(key,JSON.stringify(saved));
    });
  });
});
(function(){
  const pop=document.createElement('div');
  pop.id='fv-pop';
  const img=document.createElement('img');
  pop.appendChild(img);
  document.body.appendChild(pop);
  const W=452,H=269,G=6;
  document.querySelectorAll('.fv').forEach(el=>{
    el.addEventListener('mouseenter',()=>{
      img.src='https://charts2.finviz.com/chart.ashx?t='+el.dataset.sym+'&ty=c&ta=1&p=d';
      const r=el.getBoundingClientRect();
      const below=window.innerHeight-r.bottom, above=r.top;
      if(below>=H+G||below>=above){
        pop.style.top=(r.bottom+G)+'px';pop.style.bottom='auto';
      } else {
        pop.style.top='auto';pop.style.bottom=(window.innerHeight-r.top+G)+'px';
      }
      let left=r.left;
      if(left+W>window.innerWidth-8)left=window.innerWidth-W-8;
      pop.style.left=Math.max(8,left)+'px';
      pop.style.display='block';
    });
    el.addEventListener('mouseleave',()=>{pop.style.display='none';});
  });
})();
"""

MAIN_F = ["symbol","score","price","analyst_rating","analyst_target","analyst_upside",
          "fundamental_score","rev_growth","eps_beat_rate","fcf_margin","bull_pct",
          "rsi","pcr","rr_25d_pct",
          "exp_date","dte","strike","moneyness","exp_move_pct","vs_em",
          "bid","ask","spread","volume","be_bid","pct_be_bid","open_int",
          "iv_rank","iv","hv30","delta","theta","ret","ann_rtn","profit_prob","ma200_pct"]
MAIN_H = ["Symbol","Score","Price~","Rating","Target","Upside%",
          "FundScore","RevGrow%","BeatRate%","FCF%","ST Bull%",
          "RSI","PCR","Skew%",
          "Exp Date","DTE","Strike","Mness%","EM%","vs EM%",
          "Bid","Ask","Spread","Vol","BE(Bid)","%BE","OI",
          "IVR","IV%","HV30%","Delta","θ/day","Ret%","AnnRtn%","PProb%","MA200%"]

IV_F = ["symbol","price","analyst_rating","analyst_target","analyst_upside",
        "exp_date","strike","exp_move_pct","vs_em","bid","ask","spread",
        "volume","iv_rank","iv","hv30","ann_rtn","theta","profit_prob"]
IV_H = ["Symbol","Price~","Rating","Target","Upside%",
        "Exp Date","Strike","EM%","vs EM%","Bid","Ask","Spread",
        "Vol","IVR","IV%","HV30%","AnnRtn%","θ/day","PProb%"]

COL_DESC = {
    "symbol":            "Ticker — click for Finviz, hover for chart",
    "score":             "Composite ranking score (probability, return, EM buffer, MA200, fundamentals)",
    "price":             "Underlying spot price (15-20 min delayed)",
    "analyst_rating":    "Mean analyst recommendation (BUY / HOLD / SELL) with N analysts",
    "analyst_target":    "Mean analyst 12-month price target",
    "analyst_upside":    "% upside from spot to analyst target",
    "fundamental_score": "0–100 score from EPS beat rate, revenue growth, FCF margin, debt/equity (financialdatasets.ai)",
    "rev_growth":        "Year-over-year revenue growth %",
    "eps_beat_rate":     "% of last 8 quarters where EPS beat estimate",
    "fcf_margin":        "Free cash flow / revenue %",
    "bull_pct":          "StockTwits bullish sentiment % (≥60 bullish, ≤40 bearish)",
    "rsi":               "14-day Wilder RSI; ≥70 overbought, ≤30 oversold",
    "pcr":               "Put/Call ratio from option chain volume; ≥1.5 bearish, ≤0.5 bullish",
    "rr_25d_pct":        "25-delta Risk Reversal % — put skew vs calls; high = puts expensive (fear)",
    "exp_date":          "Option expiration date",
    "dte":               "Days to expiration",
    "strike":            "Put strike price",
    "moneyness":         "(strike − spot) / spot %  — negative = OTM put",
    "exp_move_pct":      "Expected move % until expiry (from ATM straddle)",
    "vs_em":             "Strike distance vs expected move (>100% = strike beyond 1σ implied move)",
    "bid":               "Option bid price",
    "ask":               "Option ask price",
    "spread":            "Bid-ask spread (absolute $)",
    "volume":            "Today's option contract volume",
    "be_bid":            "Break-even price using bid (strike − bid)",
    "pct_be_bid":        "% spot is above break-even (cushion)",
    "open_int":          "Open interest (outstanding contracts)",
    "iv_rank":           "IV Rank 0–100: where current IV sits in 52-week range",
    "iv":                "Implied volatility % (annualized)",
    "hv30":              "30-day historical (realized) volatility %",
    "delta":             "Option delta (≈ probability of finishing ITM)",
    "theta":             "Theta — $ premium decay per day",
    "ret":               "One-period return: premium / (strike × 100)",
    "ann_rtn":           "Annualized return %",
    "profit_prob":       "Probability of expiring OTM (Black-Scholes)",
    "ma200_pct":         "% spot is above/below 200-day moving average",
}


def _score_bg(score: float, lo: float, hi: float) -> str:
    """Row background: dark → green by normalized score."""
    try:
        t = 0.0 if hi <= lo else max(0.0, min(1.0, (float(score) - lo) / (hi - lo)))
    except (TypeError, ValueError):
        t = 0.0
    return f"rgb({int(17+t*4)},{int(24+t*104)},{int(39+t*22)})"


def _col_toolbar(fields: list, headers: list, table_id: str,
                 visible: set[str]) -> str:
    """Render checkbox grid that toggles column visibility for a table."""
    parts = [f'<div class="col-toggles" data-target="{table_id}">']
    for f, h in zip(fields, headers):
        desc  = _he.escape(COL_DESC.get(f, ""))
        label = _he.escape(h)
        chk   = " checked" if f in visible else ""
        parts.append(
            f'<label title="{desc}">'
            f'<input type="checkbox" data-col="{f}"{chk}>{label}</label>'
        )
    parts.append('</div>')
    return "".join(parts)


def _html_table(subset: pd.DataFrame, fields: list, headers: list,
                table_id: str, s_lo: float, s_hi: float,
                default_visible: list | None = None) -> str:
    """Return a column-toolbar + sortable HTML <table> string.

    default_visible: list of fields shown initially (others hidden but toggleable).
    None = all fields visible.
    """
    if subset.empty:
        return '<p class="empty">No results in this category.</p>'

    visible = set(fields if default_visible is None else default_visible)
    toolbar = _col_toolbar(fields, headers, table_id, visible)
    head = ("<thead><tr>"
            + "".join(
                f'<th data-col="{f}"'
                + ('' if f in visible else ' style="display:none"')
                + f'>{_he.escape(h)}</th>'
                for f, h in zip(fields, headers))
            + "</tr></thead>")
    _hide = lambda f: '' if f in visible else ' style="display:none"'
    rows = []
    for rec in subset.to_dict("records"):
        is_earn = bool(rec.get("earnings_date", ""))
        bg      = _score_bg(rec.get("score", float("nan")), s_lo, s_hi)
        cells   = []
        for f in fields:
            v = rec.get(f)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append(f'<td data-col="{f}"{_hide(f)} data-v="-9999">—</td>')
            elif f == "symbol":
                raw = str(v).replace(" [!]", "")
                s   = _he.escape(raw)
                tag = f'{s} [!]' if is_earn else s
                cls = ' class="earn"' if is_earn else ""
                cells.append(
                    f'<td data-v="{s}"{cls}>'
                    f'<span class="fv" data-sym="{s}">'
                    f'<a href="https://finviz.com/quote.ashx?t={raw}" target="_blank">{tag}</a>'
                    f'</span></td>'
                )
            elif f == "analyst_rating":
                raw  = str(v)
                abbr = RATING_ABBREV.get(raw, raw)
                n    = int(rec.get("analyst_num") or 0)
                disp = _he.escape(f"{abbr} ({n})" if abbr and n > 0 else abbr or "—")
                cls  = RATING_CLASS.get(raw, "")
                ca   = f' class="{cls}"' if cls else ""
                cells.append(f'<td data-col="{f}"{_hide(f)} data-v="{_he.escape(raw)}"{ca}>{disp}</td>')
            elif f == "bull_pct":
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="-9999">—</td>')
                else:
                    pct = float(v)
                    cls = "s-bull" if pct >= 60 else ("s-bear" if pct <= 40 else "s-neu")
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="{pct}" class="{cls}">{pct}%</td>')
            elif f == "rsi":
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="-9999">—</td>')
                else:
                    r = float(v)
                    cls = "rsi-ob" if r >= 70 else ("rsi-os" if r <= 30 else "")
                    ca  = f' class="{cls}"' if cls else ""
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="{r}"{ca}>{r}</td>')
            elif f == "pcr":
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="-9999">—</td>')
                else:
                    r = float(v)
                    cls = "pcr-hi" if r >= 1.5 else ("pcr-lo" if r <= 0.5 else "")
                    ca  = f' class="{cls}"' if cls else ""
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="{r}"{ca}>{r}</td>')
            elif f == "rr_25d_pct":
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="-9999">—</td>')
                else:
                    r = float(v)
                    cls = "skew-steep" if r >= 15 else ("skew-inv" if r < 0 else "")
                    ca  = f' class="{cls}"' if cls else ""
                    cells.append(f'<td data-col="{f}"{_hide(f)} data-v="{r}"{ca}>{r}%</td>')
            elif isinstance(v, int):
                cells.append(f'<td data-col="{f}"{_hide(f)} data-v="{v}">{v:,}</td>')
            elif isinstance(v, float):
                cells.append(f'<td data-col="{f}"{_hide(f)} data-v="{v}">{v}</td>')
            else:
                cells.append(f'<td data-col="{f}"{_hide(f)} data-v="-9999">{_he.escape(str(v))}</td>')
        rows.append(f'<tr style="background:{bg}">{"".join(cells)}</tr>')

    body = "<tbody>" + "".join(rows) + "</tbody>"
    return (toolbar
            + f'<div class="wrap"><table id="{table_id}">{head}{body}</table></div>')


def write_html(df: pd.DataFrame, config_str: str, failed: list,
               ai_text: str | None = None, ai_top_n: int = 10,
               html_out: str = "csp_scan.html",
               nav_html: str = "") -> None:
    """Write all scanner results to a self-contained dark-themed sortable HTML page."""
    scores = df["score"].dropna()
    s_lo, s_hi = float(scores.min()), float(scores.max())

    no_earn  = df["earnings_date"].eq("")
    high_ivr = df["iv_rank"] >= 50
    iv_gt_hv = df["iv"] > df["hv30"]

    parts: list[str] = [
        nav_html,
        f'<h1>Short Put Scanner</h1>'
        f'<p class="meta">{_he.escape(config_str)}'
        f'<br>Generated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>'
    ]

    if ai_text:
        parts.append(
            f'<h2>AI Analysis (top {ai_top_n} by score)</h2>'
            f'<div class="ai-box">{_md_to_html(ai_text)}</div>'
        )

    earn_df = df[df["earnings_date"].ne("")]
    if not earn_df.empty:
        parts.append("<h2>Earnings Within Expiry Window</h2>")
        parts.append(_html_table(earn_df.sort_values("score", ascending=False),
                                 MAIN_F, MAIN_H, "tbl-earn", s_lo, s_hi))

    parts.append("<h2>All Results</h2>")
    parts.append(_html_table(df.sort_values("ann_rtn", ascending=False),
                             MAIN_F, MAIN_H, "tbl-main", s_lo, s_hi))

    for title, mask, tid in [
        ("HIGH IVR + IV > HV30 — Best Premium Candidates",
         high_ivr &  iv_gt_hv & no_earn, "tbl-both"),
        ("HIGH IVR (>=50) Only",
         high_ivr & ~iv_gt_hv & no_earn, "tbl-ivr"),
        ("IV > HV30 Only",
        ~high_ivr &  iv_gt_hv & no_earn, "tbl-hv"),
    ]:
        parts.append(f"<h2>{_he.escape(title)}</h2>")
        parts.append(_html_table(df[mask].sort_values("score", ascending=False),
                                 MAIN_F, MAIN_H, tid, s_lo, s_hi,
                                 default_visible=IV_F))

    if failed:
        fs = ", ".join(_he.escape(s) for s in failed)
        parts.append(f'<p class="failed">[!] {len(failed)} tickers skipped (rate limited): {fs}</p>')

    page = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Short Put Scanner</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        + "".join(parts)
        + f"<script>{_JS}</script></body></html>"
    )

    with open(html_out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Saved HTML  to {html_out}")
