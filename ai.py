"""Claude AI analysis of CSP scan candidates."""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import PutRow

_DEFAULT_MODEL = "claude-opus-4-7"


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(v, suffix: str = "", fallback: str = "n/a") -> str:
    try:
        f = float(v)
        return f"{f}{suffix}" if not math.isnan(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _fmt_billions(v) -> str:
    """Format a large number as $XB / $XM for readability."""
    try:
        n = float(v)
        if math.isnan(n):
            return "n/a"
        if abs(n) >= 1e9:
            return f"${n/1e9:.1f}B"
        if abs(n) >= 1e6:
            return f"${n/1e6:.1f}M"
        return f"${n:.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _md_to_html(text: str) -> str:
    """Minimal Markdown → HTML: bold, headers, paragraphs."""
    import html as _he
    import re
    t = _he.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"^#{1,3} (.+)$", r"<h3>\1</h3>", t, flags=re.MULTILINE)
    t = t.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{t}</p>"


def _format_fund_raw(sym: str, raw: dict) -> str:
    """Format raw financialdatasets.ai data as compact text for the AI prompt."""
    lines: list[str] = []

    income = (raw.get("income") or [])[:3]
    if income:
        lines.append("  Annual P&L (newest first):")
        for s in income:
            date = str(s.get("calendar_date") or s.get("period") or "")[:7]
            rev  = _fmt_billions(s.get("revenue") or s.get("total_revenue"))
            oi   = _fmt_billions(s.get("operating_income"))
            ni   = _fmt_billions(s.get("net_income"))
            eps  = s.get("eps_diluted") or s.get("eps")
            eps_s = f"${float(eps):.2f}" if eps is not None else "n/a"
            lines.append(f"    {date}: rev={rev}  op_inc={oi}  net={ni}  EPS={eps_s}")

    earnings = (raw.get("earnings") or [])[:8]
    if earnings:
        lines.append("  Quarterly EPS (actual vs estimate):")
        for e in earnings:
            period = str(e.get("period") or e.get("calendar_date") or "")[:7]
            act    = e.get("actual_eps")
            est    = e.get("estimated_eps")
            if act is not None and est is not None:
                act_f, est_f = float(act), float(est)
                marker   = "✓" if act_f >= est_f else "✗"
                surprise = (act_f - est_f) / abs(est_f) * 100 if est_f != 0 else 0
                lines.append(f"    {period}: actual={act_f:.2f}  est={est_f:.2f}  "
                              f"({surprise:+.1f}%) {marker}")

    balance = (raw.get("balance") or [])[:1]
    if balance:
        bs   = balance[0]
        debt = _fmt_billions(bs.get("total_debt") or bs.get("long_term_debt"))
        cash = _fmt_billions(bs.get("cash_and_equivalents")
                             or bs.get("cash_and_short_term_investments"))
        eq   = _fmt_billions(bs.get("total_equity") or bs.get("shareholders_equity")
                             or bs.get("total_stockholders_equity"))
        lines.append(f"  Balance sheet (latest): debt={debt}  cash={cash}  equity={eq}")

    cashflow = (raw.get("cashflow") or [])[:2]
    if cashflow:
        lines.append("  Cash flow (annual):")
        for cf in cashflow:
            date  = str(cf.get("calendar_date") or cf.get("period") or "")[:7]
            ocf   = _fmt_billions(cf.get("operating_cash_flow") or cf.get("cash_from_operations"))
            fcf   = _fmt_billions(cf.get("free_cash_flow"))
            capex = _fmt_billions(cf.get("capital_expenditure") or cf.get("capex"))
            lines.append(f"    {date}: op_cf={ocf}  capex={capex}  fcf={fcf}")

    news = (raw.get("news") or [])[:5]
    if news:
        lines.append("  Recent news:")
        for item in news:
            date  = str(item.get("published_at") or item.get("date") or "")[:10]
            title = item.get("title") or item.get("headline") or ""
            if title:
                lines.append(f"    {date}: {title}")

    return "\n".join(lines) if lines else "  (fundamental data unavailable)"


# ── Main analysis function ────────────────────────────────────────────────────

def ai_analysis(top_rows: list[PutRow], fund_raw: dict | None = None,
                model: str = _DEFAULT_MODEL) -> str | None:
    """Send top N put candidates to Claude for qualitative analysis.

    Requires: pip install anthropic  +  ANTHROPIC_API_KEY env var.
    Returns the analysis text, or None if unavailable.
    The caller decides whether to invoke this (ENABLE_AI_ANALYSIS check).
    """
    try:
        import anthropic
    except ImportError:
        print("[AI] 'anthropic' not installed — run: pip install anthropic")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[AI] ANTHROPIC_API_KEY not set — skipping AI analysis")
        return None

    lines = [
        "You are a quantitative options analyst. Analyze these cash-secured put (short put) "
        "candidates for a retail trader. Data is from a systematic scanner using Yahoo Finance "
        "(15–20 min delayed). Fundamental data (P&L, EPS history, balance sheet, cash flow) "
        "is sourced from SEC filings via financialdatasets.ai.\n\n"
        "Candidates (sorted by composite score, best first):\n\n"
    ]
    for i, r in enumerate(top_rows, 1):
        earn       = f"⚠ Earnings {r.earnings_date} within window" if r.earnings_date else "No earnings in window"
        rating_str = f"{r.analyst_rating} ({r.analyst_num} analysts)" if r.analyst_rating else "n/a"
        raw        = (fund_raw or {}).get(r.symbol, {})
        fund_block   = _format_fund_raw(r.symbol, raw) if raw else ""
        sent_str     = (f"{r.bull_pct:.0f}% bullish" if not math.isnan(r.bull_pct)
                        else "n/a")
        rsi_str  = _fmt(r.rsi, "")
        pcr_str  = _fmt(r.pcr, "")
        skew_str = _fmt(r.rr_25d_pct, "%") if not math.isnan(r.rr_25d_pct) else "n/a"
        lines.append(
            f"{i}. {r.symbol}  —  Put ${r.strike}  exp {r.exp_date}  DTE {r.dte}\n"
            f"   Price ${r.price}  |  Bid ${r.bid}  |  Ann return {r.ann_rtn}%  |  Profit prob {r.profit_prob}%\n"
            f"   IV {r.iv}%  |  HV30 {r.hv30}%  |  IVR {r.iv_rank}  |  Delta {r.delta}\n"
            f"   Moneyness {r.moneyness:+.1f}%  |  vs EM {r.vs_em:.0f}%  |  MA200 {r.ma200_pct:+.1f}%\n"
            f"   RSI(14): {rsi_str}  |  PCR: {pcr_str}  |  Skew(25Δ RR): {skew_str}  |  Analyst: {rating_str}  |  Upside {_fmt(r.analyst_upside, '%')}\n"
            f"   StockTwits: {sent_str}  |  Fundamental score: {_fmt(r.fundamental_score)}  |  {earn}\n"
            + (f"{fund_block}\n" if fund_block else "")
            + "\n"
        )
    lines.append(
        "For each position provide:\n"
        "- 2-3 sentences: key attractiveness factors + main risk(s). "
        "Reference specific numbers from the fundamental data where relevant "
        "(e.g. revenue trend, EPS beat consistency, FCF strength, debt level).\n"
        "- Risk: Low / Medium / High\n"
        "- Verdict: Recommended / Neutral / Avoid\n\n"
        "End with a 2-sentence overall market observation implied by these results.\n"
        "Be concise and actionable. No generic disclaimers."
    )

    print(f"[AI] Requesting analysis of top {len(top_rows)} candidates from {model}...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1800,
            messages=[{"role": "user", "content": "".join(lines)}],
        )
        return msg.content[0].text
    except Exception as e:
        print(f"[AI] Error: {e}")
        return None
