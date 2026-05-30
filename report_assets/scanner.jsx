/* Short Put Scanner — trader-grade UI (generated from Claude Design handoff) */
const { useState, useMemo, useEffect, useRef, useLayoutEffect } = React;

const TAB_META = [
  { id: "earnings", label: "Earnings risk", hint: "Earnings inside expiry window" },
  { id: "all",      label: "All results",   hint: "Full scan output" },
  { id: "premium",  label: "Premium",       hint: "High IVR + IV > HV30" },
  { id: "highIvr",  label: "High IVR",      hint: "IVR ≥ 50" },
  { id: "ivOverHv", label: "IV > HV30",     hint: "IV richer than realized vol" },
];

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

const fmt = {
  num:    (v, d = 2) => v == null ? "—" : (typeof v === "number" ? v.toFixed(d) : v),
  int:    (v)        => v == null ? "—" : v.toLocaleString("en-US"),
  date:   (s) => {
    if (!s) return "—";
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
    if (!m) return s;
    const yy = m[1].slice(-2);
    return `${MONTHS[parseInt(m[2], 10) - 1]} ${parseInt(m[3], 10)}'${yy}`;
  },
  signed: (v, d = 1) => v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d),
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function ScoreCell({ score }) {
  if (score == null) return <span className="muted">—</span>;
  const pct = clamp(score, 0, 100);
  return (
    <div className="score-pill">
      <span className="score-num">{score.toFixed(1)}</span>
      <span className="score-bar-track">
        <span className="score-bar-fill" style={{ width: pct + "%" }} />
      </span>
    </div>
  );
}

function RatingPill({ rating, count }) {
  const key = (rating || "none").replace(/[^A-Z_]/gi, "") || "none";
  const cls = "rating-pill rating-" + key;
  const label =
    rating === "STR_BUY" ? "Str Buy" :
    rating === "BUY"     ? "Buy" :
    rating === "HOLD"    ? "Hold" :
    rating === "UNDP"    ? "Undp" :
    rating === "SELL"    ? "Sell" :
    "—";
  return (
    <span className={cls}>
      <span>{label}</span>
      {count != null && <span className="rating-count">{count}</span>}
    </span>
  );
}

function BarCell({ value, max = 100, kind, fmtVal, neg = false }) {
  if (value == null) return <span className="muted">—</span>;
  const pct = clamp(Math.abs(value) / max * 100, 0, 100);
  return (
    <div className={"bar-cell bar-" + kind}>
      <span className={"bar-val " + (neg && value < 0 ? "neg" : "")}>
        {fmtVal ? fmtVal(value) : fmt.num(value, 1)}
      </span>
      <span className="bar-bg">
        <span className="bar-fg" style={{ width: pct + "%" }} />
      </span>
    </div>
  );
}

function IvHvCell({ iv, hv }) {
  if (iv == null) return <span className="muted">—</span>;
  const over = hv != null && iv > hv;
  const ratio = hv ? iv / hv : 1;
  return (
    <span className={"iv-pair " + (over ? "over" : "under")}>
      <span>{iv.toFixed(1)}</span>
      <span className="arrow">{over ? "▲" : "▼"}</span>
      <span className="muted">{ratio.toFixed(1)}×</span>
    </span>
  );
}

function Signed({ v, d = 1, suffix = "" }) {
  if (v == null) return <span className="muted">—</span>;
  const cls = v > 0 ? "pos" : v < 0 ? "neg" : "muted";
  return <span className={cls}>{(v > 0 ? "+" : "") + v.toFixed(d)}{suffix}</span>;
}

function PayoffChart({ row }) {
  const W = 460, H = 150, PAD_L = 36, PAD_R = 14, PAD_T = 16, PAD_B = 26;
  const cw = W - PAD_L - PAD_R, ch = H - PAD_T - PAD_B;
  const strike = row.strike, prem = row.bid, be = row.beBid, price = row.price;
  if (strike == null || prem == null || be == null || price == null) {
    return <div className="muted" style={{padding:"20px"}}>Insufficient data for payoff chart.</div>;
  }
  const xMin = strike * 0.7;
  const xMax = strike * 1.15;
  const xs = [];
  for (let i = 0; i <= 60; i++) xs.push(xMin + (xMax - xMin) * i / 60);
  const pls = xs.map(s => Math.min(0, s - strike) + prem);
  const yMin = Math.min(...pls, -prem * 3);
  const yMax = Math.max(prem * 1.2, 0.2);
  const sx = v => PAD_L + (v - xMin) / (xMax - xMin) * cw;
  const sy = v => PAD_T + (1 - (v - yMin) / (yMax - yMin)) * ch;
  const path = pls.map((y, i) => `${i ? "L" : "M"}${sx(xs[i]).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");
  const zeroY = sy(0);
  const negPts = xs.map((x, i) => [x, Math.min(0, pls[i])]);
  const areaPath = `M${sx(xs[0])},${zeroY} ` +
    negPts.map(([x, y]) => `L${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ") +
    ` L${sx(xs[xs.length-1])},${zeroY} Z`;
  const profitArea = `M${sx(xs[0])},${zeroY} ` +
    xs.map((x, i) => `L${sx(x).toFixed(1)},${sy(Math.max(0, pls[i])).toFixed(1)}`).join(" ") +
    ` L${sx(xs[xs.length-1])},${zeroY} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="payoff" preserveAspectRatio="xMidYMid meet">
      <line x1={PAD_L} x2={W-PAD_R} y1={zeroY} y2={zeroY} stroke="var(--border)" strokeWidth="1" />
      <text x={PAD_L} y={H-8} fill="var(--text-3)" fontSize="9" fontFamily="var(--mono)">{xMin.toFixed(0)}</text>
      <text x={W-PAD_R} y={H-8} fill="var(--text-3)" fontSize="9" fontFamily="var(--mono)" textAnchor="end">{xMax.toFixed(0)}</text>
      <text x={4} y={sy(0)+3} fill="var(--text-3)" fontSize="9" fontFamily="var(--mono)">$0</text>
      <text x={4} y={sy(prem)+3} fill="var(--accent)" fontSize="9" fontFamily="var(--mono)">+{prem.toFixed(2)}</text>
      <path d={profitArea} fill="color-mix(in oklab, var(--accent) 18%, transparent)" />
      <path d={areaPath} fill="color-mix(in oklab, var(--danger) 16%, transparent)" />
      <path d={path} fill="none" stroke="var(--text)" strokeWidth="1.4" />
      <line x1={sx(strike)} x2={sx(strike)} y1={PAD_T} y2={H-PAD_B} stroke="var(--text-3)" strokeDasharray="2,3" strokeWidth="1" />
      <line x1={sx(be)} x2={sx(be)} y1={PAD_T} y2={H-PAD_B} stroke="var(--accent-2)" strokeDasharray="2,3" strokeWidth="1" />
      <line x1={sx(price)} x2={sx(price)} y1={PAD_T} y2={H-PAD_B} stroke="var(--info)" strokeWidth="1" />
      {(() => {
        const labels = [
          { x: sx(strike), text: "Strike $" + strike,         fill: "var(--text-2)", row: "top" },
          { x: sx(price),  text: "Spot $" + price.toFixed(0), fill: "var(--info)",   row: "top" },
          { x: sx(be),     text: "BE $" + be.toFixed(2),      fill: "var(--accent-2)",row: "bot" },
        ];
        const MIN_GAP = 32;
        const placed = [];
        return labels.map((lbl, i) => {
          let x = lbl.x, anchor = "middle";
          for (const p of placed) {
            if (p.row !== lbl.row) continue;
            const dx = x - p.x;
            if (Math.abs(dx) < MIN_GAP) {
              if (dx >= 0) { anchor = "start"; x = p.x + MIN_GAP/2 + 2; }
              else         { anchor = "end";   x = p.x - MIN_GAP/2 - 2; }
              break;
            }
          }
          placed.push({ x, row: lbl.row });
          const y = lbl.row === "top" ? PAD_T - 4 : H - PAD_B + 12;
          return (
            <text key={i} x={x} y={y} fill={lbl.fill} fontSize="9" fontFamily="var(--mono)" textAnchor={anchor}>
              {lbl.text}
            </text>
          );
        });
      })()}
    </svg>
  );
}

const N = (v, d = 2, suffix = "") => v == null ? "—" : v.toFixed(d) + suffix;

function Drawer({ row, onClose }) {
  if (!row) return (
    <>
      <div className="drawer-backdrop" />
      <aside className="drawer" aria-hidden="true" />
    </>
  );
  const collat = row.strike != null ? row.strike * 100 : null;
  const credit = row.bid    != null ? row.bid * 100    : null;
  return (
    <>
      <div className="drawer-backdrop open" onClick={onClose} />
      <aside className="drawer open" role="dialog" aria-label={"Detail " + row.sym}>
        <div className="drawer-head">
          <div className="drawer-title">
            <span className="sym">{row.sym}</span>
            {row.name && <span className="cname" title={row.name}>{row.name}</span>}
            {row.price != null && <span className="price">${row.price.toFixed(2)}</span>}
            <RatingPill rating={row.rating} count={row.ratingN != null ? `${row.ratingN} an` : null} />
            {row.earn && <span className="earn-badge">EARN</span>}
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="drawer-body">
          <div className="section-h">Trade — sell {row.dte}-day {row.strike} put</div>
          <div className="kvgrid">
            <div className="kv"><span className="k">Credit (bid)</span><span className="v">{credit == null ? "—" : "$" + credit.toFixed(0)}</span></div>
            <div className="kv"><span className="k">Collateral</span><span className="v">{collat == null ? "—" : "$" + collat.toFixed(0)}</span></div>
            <div className="kv"><span className="k">Return</span><span className="v pos">{N(row.retPct, 2, "%")}</span></div>
            <div className="kv"><span className="k">Annualized</span><span className="v pos">{N(row.annRtn, 1, "%")}</span></div>
            <div className="kv"><span className="k">P(profit)</span><span className="v">{N(row.pProb, 1, "%")}</span></div>
            <div className="kv"><span className="k">Breakeven</span><span className="v">{row.beBid == null ? "—" : "$" + row.beBid.toFixed(2) + (row.bePct != null ? ` (${row.bePct.toFixed(1)}%)` : "")}</span></div>
          </div>

          <div className="section-h">Finviz chart</div>
          <a className="finviz-chart" href={`https://finviz.com/quote.ashx?t=${row.sym}`} target="_blank" rel="noreferrer">
            <img src={`https://charts2.finviz.com/chart.ashx?t=${row.sym}&ty=c&ta=1&p=d`} alt={`${row.sym} chart`} />
          </a>

          <div className="section-h">Payoff at expiry</div>
          <PayoffChart row={row} />

          <div className="section-h">Greeks &amp; pricing</div>
          <div className="kvgrid">
            <div className="kv"><span className="k">Bid / Ask</span><span className="v">{N(row.bid)} / {N(row.ask)}</span></div>
            <div className="kv"><span className="k">Spread</span><span className="v">{row.spread == null ? "—" : "$" + row.spread.toFixed(2)}</span></div>
            <div className="kv"><span className="k">Delta</span><span className="v">{N(row.delta, 3)}</span></div>
            <div className="kv"><span className="k">θ / day</span><span className="v">{N(row.theta, 3)}</span></div>
            <div className="kv"><span className="k">IV</span><span className="v">{N(row.iv, 1, "%")}</span></div>
            <div className="kv"><span className="k">HV30</span><span className="v">{N(row.hv30, 1, "%")}</span></div>
            <div className="kv"><span className="k">IVR</span><span className="v">{N(row.ivr, 1)}</span></div>
            <div className="kv"><span className="k">Vol / OI</span><span className="v">{fmt.int(row.vol)} / {fmt.int(row.oi)}</span></div>
          </div>

          <div className="section-h">Underlying</div>
          <div className="kvgrid">
            <div className="kv"><span className="k">Target</span><span className="v">{row.target == null ? "—" : "$" + row.target.toFixed(2)}</span></div>
            <div className="kv"><span className="k">Upside</span><span className="v"><Signed v={row.upsidePct} suffix="%" /></span></div>
            <div className="kv"><span className="k">RSI</span><span className="v">{N(row.rsi, 1)}</span></div>
            <div className="kv"><span className="k">PCR</span><span className="v">{N(row.pcr, 2)}</span></div>
            <div className="kv"><span className="k">vs MA200</span><span className="v"><Signed v={row.ma200} suffix="%" /></span></div>
            <div className="kv"><span className="k">Skew</span><span className="v">{N(row.skew, 2, "%")}</span></div>
          </div>

          <div className="section-h">External</div>
          <a className="linkout" href={`https://finviz.com/quote.ashx?t=${row.sym}`} target="_blank" rel="noreferrer">
            View {row.sym} on Finviz ↗
          </a>
        </div>
      </aside>
    </>
  );
}

const COLUMNS = [
  { key: "sym",      label: "Symbol",  sticky: true,                         group: "core", always: true },
  { key: "score",    label: "Score",   kind: "score",                        group: "core" },
  { key: "rating",   label: "Rating",  kind: "rating",                       group: "core" },
  { key: "price",    label: "Price",   fmt: v => "$" + v.toFixed(2),         group: "core" },

  { key: "target",   label: "Target",   fmt: v => "$" + v.toFixed(2),         group: "fund" },
  { key: "upsidePct",label: "Upside",   kind: "signed", suffix: "%",          group: "fund" },
  { key: "fundScore",label: "FundScore",fmt: v => v.toFixed(0),               group: "fund" },
  { key: "revGrow",  label: "RevGrow%", kind: "signed", suffix: "%",          group: "fund" },
  { key: "beatRate", label: "BeatRate%",fmt: v => v.toFixed(0) + "%",         group: "fund" },
  { key: "fcf",      label: "FCF%",     kind: "signed", suffix: "%",          group: "fund" },
  { key: "stBull",   label: "ST Bull%", kind: "stbull",                       group: "fund" },
  { key: "rsi",      label: "RSI",      kind: "rsi",                          group: "fund" },
  { key: "pcr",      label: "PCR",      kind: "pcr",                          group: "fund" },
  { key: "skew",     label: "Skew%",    kind: "skew",                         group: "fund", defaultHidden: true },
  { key: "ma200",    label: "MA200",    kind: "signed", suffix: "%",          group: "fund" },

  { key: "exp",      label: "Exp",     kind: "exp",                          group: "opt" },
  { key: "dte",      label: "DTE",                                            group: "opt" },
  { key: "strike",   label: "Strike",  fmt: v => "$" + v.toFixed(0),         group: "opt" },
  { key: "mnessPct", label: "Mness",   kind: "signed", suffix: "%",          group: "opt" },
  { key: "emPct",    label: "EM%",     fmt: v => v.toFixed(2) + "%",         group: "opt", defaultHidden: true },
  { key: "vsEm",     label: "vs EM",   fmt: v => v.toFixed(1),               group: "opt", defaultHidden: true },
  { key: "bid",      label: "Bid",     fmt: v => v.toFixed(2),               group: "opt" },
  { key: "ask",      label: "Ask",     fmt: v => v.toFixed(2),               group: "opt", defaultHidden: true },
  { key: "spread",   label: "Spread",  fmt: v => "$" + v.toFixed(2),         group: "opt" },
  { key: "vol",      label: "Vol",     fmt: v => fmt.int(v),                 group: "opt" },
  { key: "oi",       label: "OI",      fmt: v => fmt.int(v),                 group: "opt" },
  { key: "beBid",    label: "BE",      fmt: v => "$" + v.toFixed(2),         group: "opt", defaultHidden: true },

  { key: "ivr",      label: "IVR",     kind: "bar", barKind: "ivr",  max: 100, group: "vol" },
  { key: "iv_hv",    label: "IV vs HV30", kind: "ivhv",                       group: "vol" },
  { key: "delta",    label: "Δ", fmt: v => v.toFixed(3),                group: "vol" },
  { key: "theta",    label: "θ/d", fmt: v => v.toFixed(3),              group: "vol" },

  { key: "retPct",   label: "Ret%",    kind: "bar", barKind: "ret",  max: 6,   group: "ret" },
  { key: "annRtn",   label: "AnnRtn%", kind: "bar", barKind: "rtn",  max: 60,  group: "ret" },
  { key: "pProb",    label: "PProb%",  kind: "bar", barKind: "prob", max: 100, group: "ret" },
  { key: "bePct",    label: "%BE",     fmt: v => v.toFixed(1) + "%",         group: "ret" },
];

const GROUP_LABELS = {
  core: "Identity",
  fund: "Underlying",
  opt:  "Option contract",
  vol:  "Volatility & greeks",
  ret:  "Returns",
};

const COL_DESC = {
  sym:       "Stock ticker. Click to open Finviz; hover the row for the daily chart.",
  score:     "Composite 0–100 rank: probability of profit, return, distance to expected move, MA200 trend, and fundamentals.",
  rating:    "Mean analyst rating (Str Buy / Buy / Hold / Undp / Sell) with the number of analysts covering it.",
  price:     "Underlying spot price (15–20 min delayed).",
  target:    "Mean analyst 12-month price target.",
  upsidePct: "% upside from spot to analyst target. Positive = analysts see room to run.",
  fundScore: "Fundamental score 0–100 from EPS beat rate, revenue growth, FCF margin, and debt/equity.",
  revGrow:   "Year-over-year revenue growth %.",
  beatRate:  "% of last 8 quarters where EPS beat the estimate.",
  fcf:       "Free cash flow / revenue %. Higher = stronger cash generation.",
  stBull:    "StockTwits bullish sentiment % of tagged messages. ≥60 bullish, ≤40 bearish.",
  rsi:       "14-day Wilder RSI. ≥70 overbought (caution), ≤30 oversold (danger for short put).",
  pcr:       "Put/Call volume ratio from the option chain. ≥1.5 bearish, ≤0.5 bullish.",
  skew:      "25-delta risk reversal %. High = puts expensive vs calls (fear). Negative = inverted skew.",
  ma200:     "% spot is above/below the 200-day moving average. Positive = uptrend.",
  exp:       "Option expiration date.",
  dte:       "Days to expiration.",
  strike:    "Put strike price.",
  mnessPct:  "Moneyness % = (strike − spot) / spot × 100. Negative = OTM put.",
  emPct:     "Expected move % until expiry, derived from the ATM straddle.",
  vsEm:      "Strike distance vs expected move. >100 = strike sits beyond 1σ implied move (safer).",
  bid:       "Option bid price (per share; ×100 for the contract).",
  ask:       "Option ask price.",
  spread:    "Bid–ask spread in $. Smaller = tighter liquidity.",
  vol:       "Today's option contract volume.",
  oi:        "Open interest — total contracts outstanding.",
  beBid:     "Break-even price using bid (strike − bid).",
  ivr:       "IV Rank 0–100 — where current IV sits in the 52-week range. Higher = options elevated vs own history.",
  iv_hv:     "IV vs HV30 — implied vs 30-day realized volatility. ▲ = IV richer than realized (good for sellers). Number is IV%, ratio is IV/HV30.",
  delta:     "Option delta — sensitivity to spot. For puts ~ negative probability of finishing ITM.",
  theta:     "Theta — $ premium decay per day (positive for the put seller).",
  retPct:    "One-period return = premium / strike, in %.",
  annRtn:    "Annualized return = Ret% × 365 / DTE.",
  pProb:     "Probability of expiring OTM (Black-Scholes), in %. Higher = safer trade.",
  bePct:     "% cushion between spot and break-even. Higher = more room before losing money.",
};

const DEFAULT_HIDDEN = COLUMNS.filter(c => c.defaultHidden).map(c => c.key);

const SETTINGS_DEFAULTS = { theme: "dark", density: "normal", accent: "#5ad19a" };
const ACCENTS = ["#5ad19a", "#6fb1e4", "#f0b35b", "#d97757", "#a78bfa"];

function loadSettings() {
  try {
    const raw = localStorage.getItem("csp.settings");
    if (!raw) return { ...SETTINGS_DEFAULTS };
    return { ...SETTINGS_DEFAULTS, ...JSON.parse(raw) };
  } catch (e) { return { ...SETTINGS_DEFAULTS }; }
}

const PROFILES = ["low", "medium", "high"];

function pickInitialProfile() {
  const fromWin = typeof window !== "undefined" && window.__INITIAL_PROFILE;
  if (fromWin && SCAN_DATA && SCAN_DATA[fromWin]) return fromWin;
  for (const p of ["medium", "low", "high"]) {
    if (SCAN_DATA && SCAN_DATA[p]) return p;
  }
  return Object.keys(SCAN_DATA || {})[0] || "medium";
}

function ScannerApp() {
  const [settings, setSettings] = useState(loadSettings);
  const setSetting = (k, v) => setSettings(s => {
    const n = { ...s, [k]: v };
    try { localStorage.setItem("csp.settings", JSON.stringify(n)); } catch (e) {}
    return n;
  });

  useEffect(() => { document.body.dataset.theme = settings.theme; }, [settings.theme]);
  useEffect(() => {
    document.documentElement.style.setProperty("--accent", settings.accent);
    document.documentElement.style.setProperty("--score-bar", settings.accent);
  }, [settings.accent]);

  const [profile, setProfile] = useState(pickInitialProfile);

  const block = (SCAN_DATA && SCAN_DATA[profile]) || { meta: {}, rows: [], tabs: {}, ai: null };
  const meta = block.meta || {};
  const rows = block.rows || [];
  const tabs = block.tabs || {};
  const ai   = block.ai   || null;

  const defaultTab = (tabs.earnings && tabs.earnings.length) ? "earnings" : "all";
  const [activeTab, setActiveTab] = useState(defaultTab);
  const [sortKey, setSortKey] = useState(defaultTab === "earnings" ? "score" : "annRtn");
  const [sortDir, setSortDir] = useState("desc");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);
  const hoverTimerRef = useRef(null);

  const showHover = (row, e) => {
    const cx = e.clientX, cy = e.clientY;
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = setTimeout(() => {
      const W = 452, H = 295, G = 14;
      let left = cx + G;
      let top  = cy + G;
      if (left + W > window.innerWidth  - 8) left = cx - W - G;
      if (top  + H > window.innerHeight - 8) top  = cy - H - G;
      setHover({ row, top: Math.max(8, top), left: Math.max(8, left) });
    }, 200);
  };
  const hideHover = () => {
    if (hoverTimerRef.current) { clearTimeout(hoverTimerRef.current); hoverTimerRef.current = null; }
    setHover(null);
  };
  const [hiddenCols, setHiddenCols] = useState(() => {
    try {
      const saved = localStorage.getItem("csp.hiddenCols");
      if (saved) return new Set(JSON.parse(saved));
    } catch (e) {}
    return new Set(DEFAULT_HIDDEN);
  });
  const [colOrder, setColOrder] = useState(() => {
    const defaultOrder = COLUMNS.map(c => c.key);
    try {
      const saved = JSON.parse(localStorage.getItem("csp.colOrder") || "null");
      if (Array.isArray(saved)) {
        const known = new Set(defaultOrder);
        const filtered = saved.filter(k => known.has(k));
        const missing = defaultOrder.filter(k => !filtered.includes(k));
        return [...filtered, ...missing];
      }
    } catch (e) {}
    return defaultOrder;
  });
  const [dragKey, setDragKey]     = useState(null);
  const [dropAtKey, setDropAtKey] = useState(null);
  const [colPickerOpen, setColPickerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    try { localStorage.setItem("csp.hiddenCols", JSON.stringify([...hiddenCols])); } catch (e) {}
  }, [hiddenCols]);
  useEffect(() => {
    try { localStorage.setItem("csp.colOrder", JSON.stringify(colOrder)); } catch (e) {}
  }, [colOrder]);

  const toggleCol = (key) => setHiddenCols(s => {
    const n = new Set(s);
    n.has(key) ? n.delete(key) : n.add(key);
    return n;
  });
  const showAllCols = () => setHiddenCols(new Set());
  const resetCols   = () => {
    setHiddenCols(new Set(DEFAULT_HIDDEN));
    setColOrder(COLUMNS.map(c => c.key));
  };

  const moveCol = (fromKey, toKey) => {
    if (!fromKey || !toKey || fromKey === toKey) return;
    setColOrder(order => {
      const fromIdx = order.indexOf(fromKey);
      const toIdx   = order.indexOf(toKey);
      if (fromIdx < 0 || toIdx < 0) return order;
      const out = order.slice();
      out.splice(fromIdx, 1);
      out.splice(out.indexOf(toKey) + (toIdx > fromIdx ? 1 : 0), 0, fromKey);
      return out;
    });
  };

  const colBtnRef = useRef(null);
  const setBtnRef = useRef(null);

  const colByKey = useMemo(() => {
    const m = {};
    COLUMNS.forEach(c => { m[c.key] = c; });
    return m;
  }, []);

  const visibleColumns = useMemo(
    () => colOrder
      .map(k => colByKey[k])
      .filter(c => c && (c.always || !hiddenCols.has(c.key))),
    [colOrder, colByKey, hiddenCols]
  );

  const tabSet = useMemo(() => new Set(tabs[activeTab] || []), [tabs, activeTab]);
  const filtered = useMemo(() => {
    let out = rows.filter(r => tabSet.has(r.sym));
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      out = out.filter(r => r.sym.toLowerCase().includes(q));
    }
    out = [...out].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "string") return sortDir === "asc" ? av.localeCompare(bv) : bv.localeCompare(av);
      return sortDir === "asc" ? av - bv : bv - av;
    });
    return out;
  }, [rows, tabSet, query, sortKey, sortDir]);

  const tabCounts = useMemo(() => {
    const o = {};
    Object.keys(tabs).forEach(k => o[k] = (tabs[k] || []).length);
    return o;
  }, [tabs]);

  const summary = useMemo(() => {
    const rows = filtered;
    if (!rows.length) return null;
    const ok = (v) => v != null;
    const mean = (k) => {
      const vs = rows.map(r => r[k]).filter(ok);
      return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null;
    };
    const max = (k) => {
      const vs = rows.map(r => r[k]).filter(ok);
      return vs.length ? Math.max(...vs) : null;
    };
    return {
      count: rows.length,
      avgScore: mean("score"),
      avgPProb: mean("pProb"),
      avgAnn:   mean("annRtn"),
      bestAnn:  max("annRtn"),
      avgIvr:   mean("ivr"),
    };
  }, [filtered]);

  const sortBy = (k) => {
    if (k === sortKey) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(k); setSortDir("desc"); }
  };

  const onProfile = (p) => {
    if (p === profile) return;
    if (!SCAN_DATA || !SCAN_DATA[p]) return;
    setProfile(p);
    try {
      const u = new URL(location.href);
      u.searchParams.set("profile", p);
      history.replaceState(null, "", u);
    } catch (e) {}
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">σ</div>
          <span className="brand-name">Short Put Scanner</span>
          <span className="brand-sub">{meta.version || ""}</span>
          {meta.indexHref && (
            <a className="brand-back" href={meta.indexHref} title="All scans">← All scans</a>
          )}
        </div>

        <div className="profile-switch" role="tablist" aria-label="Profile">
          {PROFILES.map(p => {
            const available = !!(SCAN_DATA && SCAN_DATA[p]);
            return (
              <button
                key={p}
                className={profile === p ? "active" : ""}
                onClick={() => onProfile(p)}
                disabled={!available}
                title={available ? `Switch to ${p}-risk report` : `${p}-risk data not in this scan`}
              >
                {p}
              </button>
            );
          })}
        </div>

        <div className="topbar-right">
          <div className="kpi">
            <span className="kpi-label">Universe</span>
            <span className="kpi-val">{meta.universe}</span>
          </div>
          <div className="kpi">
            <span className="kpi-label">Rows</span>
            <span className="kpi-val">{summary ? summary.count : 0}</span>
          </div>
          <div className="kpi">
            <span className="kpi-label">⌀ PProb</span>
            <span className="kpi-val">{summary && summary.avgPProb != null ? summary.avgPProb.toFixed(1) + "%" : "—"}</span>
          </div>
          <div className="kpi">
            <span className="kpi-label">Best Ann</span>
            <span className="kpi-val pos" style={{ color: "var(--accent)" }}>
              {summary && summary.bestAnn != null ? "+" + summary.bestAnn.toFixed(0) + "%" : "—"}
            </span>
          </div>
          <div className="timestamp">
            <span className="dot live" />
            {meta.generated}
          </div>
          <div className="settings-wrap">
            <button
              ref={setBtnRef}
              className={"settings-btn " + (settingsOpen ? "open" : "")}
              onClick={() => setSettingsOpen(v => !v)}
              title="Display settings"
              aria-label="Display settings"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h0a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </button>
            {settingsOpen && (
              <SettingsPanel
                anchorRef={setBtnRef}
                settings={settings}
                onChange={setSetting}
                onClose={() => setSettingsOpen(false)}
              />
            )}
          </div>
        </div>
      </header>

      <nav className="filterbar">
        {TAB_META.map(tab => (
          <div
            key={tab.id}
            className={"tab " + (activeTab === tab.id ? "active" : "")}
            onClick={() => setActiveTab(tab.id)}
            title={tab.hint}
          >
            <span>{tab.label}</span>
            <span className="count">{tabCounts[tab.id] || 0}</span>
          </div>
        ))}
        <div className="tab-spacer" />
        <div className="colpicker-wrap">
          <button
            ref={colBtnRef}
            className={"colpicker-btn " + (colPickerOpen ? "open" : "")}
            onClick={() => setColPickerOpen(v => !v)}
            title="Show/hide columns"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="16" rx="1" />
              <line x1="9" y1="4" x2="9" y2="20" />
              <line x1="15" y1="4" x2="15" y2="20" />
            </svg>
            <span>Columns</span>
            <span className="col-count">{visibleColumns.length - 1}/{COLUMNS.length - 1}</span>
          </button>
          {colPickerOpen && (
            <ColumnPicker
              hidden={hiddenCols}
              onToggle={toggleCol}
              onShowAll={showAllCols}
              onReset={resetCols}
              onClose={() => setColPickerOpen(false)}
              anchorRef={colBtnRef}
            />
          )}
        </div>
        <div className="search-wrap">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" />
          </svg>
          <input
            placeholder="Filter ticker…"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>
      </nav>

      <section className="criteria">
        <div className="crit-cell"><span className="crit-label">DTE</span><span className="crit-val">{meta.dteMin}–{meta.dteMax}</span></div>
        <div className="crit-cell"><span className="crit-label">|Δ|</span><span className="crit-val">{meta.deltaMin}–{meta.deltaMax}</span></div>
        <div className="crit-cell"><span className="crit-label">Strike ≤</span><span className="crit-val">${meta.strikeMax}</span></div>
        <div className="crit-cell"><span className="crit-label">Vol ≥</span><span className="crit-val">{meta.volMin}</span></div>
        <div className="crit-cell"><span className="crit-label">Spread ≤</span><span className="crit-val">{meta.spreadMaxPct}%</span></div>
        <div className="crit-cell"><span className="crit-label">Profile</span><span className="crit-val">{profile}</span></div>
        <div className="crit-cell"><span className="crit-label">Sort</span><span className="crit-val">{sortKey} {sortDir === "asc" ? "↑" : "↓"}</span></div>
      </section>

      {ai && ai.html && (
        <AiBanner ai={ai} />
      )}

      <main className={"table-wrap density-" + settings.density}>
        <table className="scan">
          <thead>
            <tr>
              {visibleColumns.map((c) => {
                const draggable = !c.sticky;
                const isDragging = dragKey === c.key;
                const isDropTarget = dropAtKey === c.key && dragKey && dragKey !== c.key;
                return (
                  <th
                    key={c.key}
                    draggable={draggable}
                    className={[
                      c.sticky ? "col-sym" : "",
                      "sortable",
                      sortKey === (c.key === "iv_hv" ? "iv" : c.key) ? "sorted" : "",
                      c.group ? "colgroup-" + c.group : "",
                      draggable ? "draggable" : "",
                      isDragging ? "dragging" : "",
                      isDropTarget ? "drop-target" : "",
                    ].join(" ")}
                    onClick={() => { if (!isDragging) sortBy(c.key === "iv_hv" ? "iv" : c.key); }}
                    title={COL_DESC[c.key] || c.label}
                    onDragStart={draggable ? (e) => {
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", c.key);
                      setDragKey(c.key);
                    } : undefined}
                    onDragOver={draggable ? (e) => {
                      if (!dragKey || dragKey === c.key) return;
                      e.preventDefault();
                      e.dataTransfer.dropEffect = "move";
                      if (dropAtKey !== c.key) setDropAtKey(c.key);
                    } : undefined}
                    onDragLeave={() => { if (dropAtKey === c.key) setDropAtKey(null); }}
                    onDrop={draggable ? (e) => {
                      e.preventDefault();
                      const from = e.dataTransfer.getData("text/plain") || dragKey;
                      moveCol(from, c.key);
                      setDragKey(null); setDropAtKey(null);
                    } : undefined}
                    onDragEnd={() => { setDragKey(null); setDropAtKey(null); }}
                  >
                    {c.label}
                    {sortKey === (c.key === "iv_hv" ? "iv" : c.key) && (
                      <span className="sort-ind">{sortDir === "asc" ? "↑" : "↓"}</span>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {filtered.map(row => (
              <tr
                key={row.sym}
                className={selected?.sym === row.sym ? "selected" : ""}
                onClick={() => { hideHover(); setSelected(row); }}
                onMouseEnter={(e) => showHover(row, e)}
                onMouseLeave={hideHover}
              >
                {visibleColumns.map(c => renderCell(c, row))}
              </tr>
            ))}
            {!filtered.length && (
              <tr><td colSpan={visibleColumns.length} style={{textAlign:"center", color:"var(--text-3)", padding: "40px"}}>No matches.</td></tr>
            )}
          </tbody>
        </table>
      </main>

      <footer className="footer">
        <span>Universe {meta.universe} · Profile {profile} · DTE {meta.dteMin}–{meta.dteMax} · |Δ| {meta.deltaMin}–{meta.deltaMax}</span>
        {meta.failed && meta.failed.length > 0 && (
          <span className="failed-note">{meta.failed.length} tickers skipped (rate limited): {meta.failed.join(", ")}</span>
        )}
        <span style={{marginLeft:"auto"}}>Click any row for trade detail.</span>
      </footer>

      <Drawer row={selected} onClose={() => setSelected(null)} />
      <HoverPopup data={hover} />
    </div>
  );
}

function HoverPopup({ data }) {
  if (!data) return null;
  const { row, top, left } = data;
  const label = row.name ? `${row.sym} — ${row.name}` : row.sym;
  return ReactDOM.createPortal(
    <div className="hover-pop" style={{ top, left }}>
      <div className="hover-pop-name" title={label}>{label}</div>
      <img
        src={`https://charts2.finviz.com/chart.ashx?t=${encodeURIComponent(row.sym)}&ty=c&ta=1&p=d`}
        alt={`${row.sym} chart`}
      />
    </div>,
    document.body
  );
}

function renderCell(c, row) {
  const v = row[c.key];
  if (c.key === "sym") {
    return (
      <td key={c.key} className="col-sym">
        <span className="sym-wrap">
          <a className="sym sym-link" href={`https://finviz.com/quote.ashx?t=${row.sym}`} target="_blank" rel="noreferrer" onClick={e => e.stopPropagation()}>{row.sym}</a>
          {row.earn && <span className="earn-badge">EARN</span>}
        </span>
      </td>
    );
  }
  if (c.kind === "score")  return <td key={c.key} className="score-cell" style={{textAlign:"left"}}><ScoreCell score={row.score} /></td>;
  if (c.kind === "rating") return <td key={c.key} style={{textAlign:"left"}}><RatingPill rating={row.rating} count={row.ratingN != null ? `${row.ratingN}` : null} /></td>;
  if (c.kind === "signed") return <td key={c.key}><Signed v={v} suffix={c.suffix || ""} /></td>;
  if (c.kind === "bar") {
    return (
      <td key={c.key}>
        <BarCell value={v} max={c.max} kind={c.barKind} fmtVal={(x) => x == null ? "—" : x.toFixed(c.barKind === "prob" ? 1 : (c.barKind === "ret" ? 2 : 1))} />
      </td>
    );
  }
  if (c.kind === "ivhv") return <td key={c.key}><IvHvCell iv={row.iv} hv={row.hv30} /></td>;
  if (c.kind === "exp")  return <td key={c.key} className="muted">{fmt.date(row.exp)}</td>;
  if (c.kind === "stbull") {
    if (v == null) return <td key={c.key} className="muted">—</td>;
    const cls = v >= 60 ? "pos" : v <= 40 ? "neg" : "muted";
    return <td key={c.key}><span className={cls}>{v.toFixed(0) + "%"}</span></td>;
  }
  if (c.kind === "rsi") {
    if (v == null) return <td key={c.key} className="muted">—</td>;
    const cls = v >= 70 ? "warn" : v <= 30 ? "neg" : "";
    return <td key={c.key}><span className={cls}>{v.toFixed(1)}</span></td>;
  }
  if (c.kind === "pcr") {
    if (v == null) return <td key={c.key} className="muted">—</td>;
    const cls = v >= 1.5 ? "warn" : v <= 0.5 ? "pos" : "";
    return <td key={c.key}><span className={cls}>{v.toFixed(2)}</span></td>;
  }
  if (c.kind === "skew") {
    if (v == null) return <td key={c.key} className="muted">—</td>;
    const cls = v >= 15 ? "warn" : v < 0 ? "pos" : "";
    return <td key={c.key}><span className={cls}>{v.toFixed(2) + "%"}</span></td>;
  }
  if (v == null) return <td key={c.key} className="muted">—</td>;
  return <td key={c.key}>{c.fmt ? c.fmt(v) : v}</td>;
}

function AiBanner({ ai }) {
  const [open, setOpen] = useState(true);
  return (
    <section className={"ai-banner " + (open ? "open" : "closed")}>
      <div className="ai-banner-head">
        <span className="ai-banner-title">AI analysis {ai.topN ? `— top ${ai.topN} by score` : ""}</span>
        <button className="ai-banner-toggle" onClick={() => setOpen(o => !o)}>
          {open ? "Hide" : "Show"}
        </button>
      </div>
      {open && <div className="ai-banner-body" dangerouslySetInnerHTML={{__html: ai.html}} />}
    </section>
  );
}

function SettingsPanel({ anchorRef, settings, onChange, onClose }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  useLayoutEffect(() => {
    if (!anchorRef.current) return;
    const compute = () => {
      const r = anchorRef.current.getBoundingClientRect();
      const W = 260;
      const left = Math.min(r.right - W, window.innerWidth - W - 8);
      setPos({ top: r.bottom + 6, left: Math.max(8, left) });
    };
    compute();
    window.addEventListener("resize", compute);
    window.addEventListener("scroll", compute, true);
    return () => {
      window.removeEventListener("resize", compute);
      window.removeEventListener("scroll", compute, true);
    };
  }, [anchorRef]);

  useEffect(() => {
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target) &&
          anchorRef.current && !anchorRef.current.contains(e.target)) onClose();
    };
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    setTimeout(() => document.addEventListener("mousedown", onDoc), 0);
    document.addEventListener("keydown", onEsc);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onEsc); };
  }, [onClose, anchorRef]);

  return ReactDOM.createPortal(
    <div className="settings-panel" ref={ref} role="dialog" style={{ top: pos.top, left: pos.left }}>
      <div className="settings-row">
        <span className="settings-label">Theme</span>
        <div className="seg">
          {["dark","light"].map(v => (
            <button key={v} className={settings.theme === v ? "on" : ""} onClick={() => onChange("theme", v)}>{v}</button>
          ))}
        </div>
      </div>
      <div className="settings-row">
        <span className="settings-label">Density</span>
        <div className="seg">
          {["compact","normal","spacious"].map(v => (
            <button key={v} className={settings.density === v ? "on" : ""} onClick={() => onChange("density", v)}>{v}</button>
          ))}
        </div>
      </div>
      <div className="settings-row">
        <span className="settings-label">Accent</span>
        <div className="seg seg-color">
          {ACCENTS.map(c => (
            <button
              key={c}
              className={"swatch " + (settings.accent === c ? "on" : "")}
              style={{ background: c }}
              onClick={() => onChange("accent", c)}
              aria-label={c}
            />
          ))}
        </div>
      </div>
    </div>,
    document.body
  );
}

function ColumnPicker({ hidden, onToggle, onShowAll, onReset, onClose, anchorRef }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  useLayoutEffect(() => {
    if (!anchorRef.current) return;
    const compute = () => {
      const r = anchorRef.current.getBoundingClientRect();
      const W = 340;
      const left = Math.min(r.right - W, window.innerWidth - W - 8);
      setPos({ top: r.bottom + 6, left: Math.max(8, left) });
    };
    compute();
    window.addEventListener("resize", compute);
    window.addEventListener("scroll", compute, true);
    return () => {
      window.removeEventListener("resize", compute);
      window.removeEventListener("scroll", compute, true);
    };
  }, [anchorRef]);

  useEffect(() => {
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target) &&
          anchorRef.current && !anchorRef.current.contains(e.target)) onClose();
    };
    const onEsc = (e) => { if (e.key === "Escape") onClose(); };
    setTimeout(() => document.addEventListener("mousedown", onDoc), 0);
    document.addEventListener("keydown", onEsc);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onEsc); };
  }, [onClose, anchorRef]);

  const groups = {};
  COLUMNS.forEach(c => {
    if (c.always) return;
    const g = c.group || "misc";
    (groups[g] = groups[g] || []).push(c);
  });
  const groupOrder = ["core", "fund", "opt", "vol", "ret"];
  return ReactDOM.createPortal(
    <div className="colpicker-panel" ref={ref} role="dialog" style={{ top: pos.top, left: pos.left }}>
      <div className="colpicker-head">
        <span className="colpicker-title">Columns</span>
        <div className="colpicker-actions">
          <button onClick={onShowAll}>Show all</button>
          <button onClick={onReset}>Reset</button>
        </div>
      </div>
      <div className="colpicker-body">
        {groupOrder.map(gKey => {
          const cols = groups[gKey];
          if (!cols) return null;
          const allHidden = cols.every(c => hidden.has(c.key));
          return (
            <div className="colpicker-group" key={gKey}>
              <div className="colpicker-group-head">
                <span>{GROUP_LABELS[gKey] || gKey}</span>
                <button
                  className="colpicker-group-toggle"
                  onClick={() => cols.forEach(c => {
                    const isHidden = hidden.has(c.key);
                    if (allHidden ? isHidden : !isHidden) onToggle(c.key);
                  })}
                >
                  {allHidden ? "Show all" : "Hide all"}
                </button>
              </div>
              <div className="colpicker-list">
                {cols.map(c => {
                  const visible = !hidden.has(c.key);
                  return (
                    <label
                      key={c.key}
                      className={"colpicker-item " + (visible ? "on" : "off")}
                      title={COL_DESC[c.key] || c.label}
                    >
                      <input type="checkbox" checked={visible} onChange={() => onToggle(c.key)} />
                      <span className="colpicker-check" aria-hidden="true">
                        {visible && (
                          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="2,6 5,9 10,3" />
                          </svg>
                        )}
                      </span>
                      <span className="colpicker-label">{c.label}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>,
    document.body
  );
}

window.ScannerApp = ScannerApp;
