"""Self-contained interactive HTML report for a single `analyze` run.

Renders a `report.json` dict into one standalone HTML file — inline CSS + SVG
charts + a little vanilla JS (hover tooltips, dark-mode toggle, table view). No
external/CDN dependencies, so the file works offline in any browser. Colors are
the dataviz validated defaults (sequential blue for the single-series charts,
the reserved status palette for the coherence gates, each with icon + label).

Charts (chosen by the job each does):
  1. stat tiles          — headline magnitudes (makespan, goodput, tail bubble…)
  2. coherence chips      — pass/fail gates as status chips (icon + label)
  3. completion CDF       — the tail: cumulative fraction of rollouts vs seconds
  4. goodput decomposition— useful vs tail-wait share of the batch wall time
  5. completion vs OSL     — scatter showing the tail is generation-bound
  6. token-domain validation — realized per-rollout OSL vs distribution targets
"""
import html
import json

# ---- palette (dataviz validated defaults); dark overrides via [data-theme=dark] ----
_CSS = """
:root{
  --surface:#fcfcfb; --panel:#ffffff; --line:#e6e5e1;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#8a897f;
  --series:#2a78d6; --series-soft:#9ec5f4; --waste:#eb6834; --grid:#eeede9;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
}
[data-theme=dark]{
  --surface:#1a1a19; --panel:#232320; --line:#3a3a36;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#8f8e83;
  --series:#3987e5; --series-soft:#1c5cab; --waste:#d95926; --grid:#2c2c29;
}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;}
main{max-width:1060px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:20px;margin:0 0 2px} h2{font-size:14px;margin:22px 0 10px;color:var(--ink2);
  font-weight:600;letter-spacing:.02em;text-transform:uppercase}
.sub{color:var(--ink2);margin:0 0 18px;font-size:13px}
.top{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
button.toggle{background:var(--panel);border:1px solid var(--line);color:var(--ink2);
  border-radius:7px;padding:6px 12px;font-size:12px;cursor:pointer}
button.toggle:hover{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px}
.tile b{display:block;font-size:22px;font-weight:650;letter-spacing:-.01em}
.tile span{color:var(--ink2);font-size:12px}
.tile small{color:var(--muted);font-size:11px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:2px}
.chip{display:inline-flex;align-items:center;gap:6px;background:var(--panel);
  border:1px solid var(--line);border-radius:999px;padding:5px 12px;font-size:12.5px}
.chip .dot{width:9px;height:9px;border-radius:50%}
.chip .ic{font-size:12px;line-height:1}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-top:10px}
.panel h3{margin:0 0 2px;font-size:13.5px} .panel p.h{color:var(--muted);font-size:11.5px;margin:0 0 8px}
svg{display:block;width:100%;height:auto;overflow:visible}
.grid-line{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:10px}
.axlabel{fill:var(--ink2);font-size:11px}
.marker{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.marker-t{fill:var(--ink2);font-size:10px;font-weight:600}
.series-line{fill:none;stroke:var(--series);stroke-width:2}
.series-fill{fill:var(--series);opacity:.10}
.dot{fill:var(--series);opacity:.72;stroke:var(--panel);stroke-width:1}
.seg-use{fill:var(--series)} .seg-waste{fill:var(--waste)}
.seg-t{fill:#fff;font-size:11px;font-weight:600}
.legend{display:flex;gap:16px;margin-top:8px;font-size:12px;color:var(--ink2)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}
th,td{text-align:left;padding:5px 10px;border-bottom:1px solid var(--line)}
th{color:var(--ink2);font-weight:600} td.n{text-align:right;font-variant-numeric:tabular-nums}
#tableview{display:none} body.showtable #charts{display:none} body.showtable #tableview{display:block}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);
  border-radius:7px;padding:6px 9px;font-size:12px;color:var(--ink);box-shadow:0 4px 14px #0002;
  opacity:0;transition:opacity .08s;z-index:9;white-space:nowrap}
.hoverpt{fill:var(--series);stroke:var(--panel);stroke-width:1.5;opacity:0}
.crosshair{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3;opacity:0}
"""


def _e(x):
    return html.escape(str(x))


def _fmt_s(v):
    if v is None:
        return "—"
    return f"{v:.0f}s" if v >= 100 else f"{v:.1f}s"


def _fmt_tok(v):
    if v is None:
        return "—"
    return f"{v/1000:.1f}k" if v >= 10000 else f"{int(v)}"


# ---------- panel 1: stat tiles ----------
def _tiles(rep):
    g = rep.get("goodput_proxy")
    t = [
        ("Makespan", _fmt_s(rep.get("makespan_s")), "slowest rollout = batch wall time"),
        ("Goodput", f"{g*100:.0f}%" if g is not None else "—", "mean completion / makespan"),
        ("Tail bubble", _fmt_s(rep.get("tail_bubble_s")), "wall time waiting past p90"),
        ("Completion p50", _fmt_s(rep.get("completion_p50_s")), "median rollout"),
        ("Completion p99", _fmt_s(rep.get("completion_p99_s")), "straggler"),
        ("Throughput", f"{rep.get('output_tok_throughput',0):.0f} tok/s", f"{rep.get('num_sessions','?')} rollouts / {rep.get('num_requests','?')} req"),
    ]
    cells = "".join(f'<div class="tile"><b>{_e(v)}</b><span>{_e(lab)}</span><br><small>{_e(sub)}</small></div>'
                    for lab, v, sub in t)
    return f'<div class="tiles">{cells}</div>'


# ---------- panel 2: coherence chips ----------
def _chips(rep):
    ap = rep.get("aiperf") or {}
    vt, vtm = rep.get("validate_token") or {}, rep.get("validate_time") or {}
    faithful = ap.get("faithful")
    errs = ap.get("error_request_count")
    items = [
        ("Faithful", faithful is True, f"{errs or 0} request errors" if faithful is not None else "no aiperf summary"),
        ("Token-domain", vt.get("passed"), "realized per-rollout OSL vs distribution"),
        ("Time-domain", vtm.get("passed"), "served tail shape vs reference"),
    ]
    out = []
    for lab, ok, sub in items:
        if ok is None:
            col, ic = "var(--muted)", "•"
        elif ok:
            col, ic = "var(--good)", "✓"
        else:
            col, ic = "var(--serious)", "!"
        out.append(f'<span class="chip" title="{_e(sub)}"><span class="dot" style="background:{col}"></span>'
                   f'<span class="ic" style="color:{col}">{ic}</span>{_e(lab)}</span>')
    return f'<div class="chips">{"".join(out)}</div>'


# ---------- small svg scaffolding ----------
def _axes(x0, y0, x1, y1):
    return (f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>'
            f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')


# ---------- panel 3: completion CDF ----------
def _cdf(rep):
    rs = rep.get("rollouts") or []
    comps = sorted(r["completion_s"] for r in rs)
    if not comps:
        return ""
    W, H, mL, mR, mT, mB = 940, 300, 46, 14, 40, 30
    x0, x1, y0, y1 = mL, W - mR, mT, H - mB
    xmax = max(comps[-1], rep.get("makespan_s") or comps[-1]) or 1
    sx = lambda v: x0 + (v / xmax) * (x1 - x0)
    sy = lambda f: y1 - f * (y1 - y0)
    n = len(comps)
    # step ECDF path
    pts = []
    for i, c in enumerate(comps):
        pts.append((sx(c), sy(i / n)))
        pts.append((sx(c), sy((i + 1) / n)))
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    fill = f"{path} L {sx(comps[-1]):.1f},{sy(0):.1f} L {sx(comps[0]):.1f},{sy(0):.1f} Z"
    # grid + y ticks (fraction)
    g = []
    for f in (0, .25, .5, .75, .9, 1.0):
        y = sy(f)
        g.append(f'<line class="grid-line" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        g.append(f'<text class="tick" x="{x0-6}" y="{y+3:.1f}" text-anchor="end">{f:.2f}</text>')
    # x ticks
    for k in range(0, 6):
        v = xmax * k / 5
        x = sx(v)
        g.append(f'<text class="tick" x="{x:.1f}" y="{y1+14}" text-anchor="middle">{_fmt_s(v)}</text>')
    # percentile + makespan markers (lines at true x; labels de-collided along the top)
    marks = [("p50", rep.get("completion_p50_s")), ("p90", rep.get("completion_p90_s")),
             ("p99", rep.get("completion_p99_s")), ("makespan", rep.get("makespan_s"))]
    marks = [(lab, sx(v)) for lab, v in marks if v is not None]
    # Stack labels into as many rows as needed so clustered markers never overlap
    # (e.g. p90/p99/makespan all landing within a percent of each other): place each
    # label on the lowest row whose last label has cleared its horizontal span.
    row_right = []          # rightmost x used, per row
    for lab, x in marks:
        g.append(f'<line class="marker" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}"/>')
        anchor = "end" if x >= x1 - 2 else "middle"
        w = 6.2 * len(lab)
        left = x - (w if anchor == "end" else w / 2)
        lvl = next((i for i, r in enumerate(row_right) if left >= r + 4), len(row_right))
        if lvl == len(row_right):
            row_right.append(0)
        row_right[lvl] = x + (0 if anchor == "end" else w / 2)
        g.append(f'<text class="marker-t" x="{min(x,x1):.1f}" y="{y0-3-lvl*12}" text-anchor="{anchor}">{lab}</text>')
    body = ("".join(g) + _axes(x0, y0, x1, y1) +
            f'<path class="series-fill" d="{fill}"/><path class="series-line" d="{path}"/>'
            f'<text class="axlabel" x="{(x0+x1)/2:.0f}" y="{H-1}" text-anchor="middle">rollout completion time</text>'
            f'<line class="crosshair" id="cdf-cx" y1="{y0}" y2="{y1}"/><circle class="hoverpt" id="cdf-hp" r="4"/>')
    svg = (f'<svg viewBox="0 0 {W} {H}" id="cdf-svg" '
           f'data-x0="{x0}" data-x1="{x1}" data-y0="{y0}" data-y1="{y1}" data-xmax="{xmax}">{body}</svg>')
    return _panel("Completion-time CDF", "cumulative fraction of rollouts finished by time t — the tail is the flat run out to the makespan", svg)


# ---------- panel 4: goodput decomposition ----------
def _goodput(rep):
    g = rep.get("goodput_proxy")
    mk = rep.get("makespan_s")
    if g is None or not mk:
        return ""
    W, H, mL, mR = 940, 92, 8, 8
    x0, x1 = mL, W - mR
    ux = x0 + g * (x1 - x0)
    y, h = 26, 34
    use_w, waste_w = ux - x0, x1 - ux
    seg = (f'<rect class="seg-use" x="{x0}" y="{y}" width="{max(0,use_w):.1f}" height="{h}" rx="4"/>'
           f'<rect class="seg-waste" x="{ux:.1f}" y="{y}" width="{max(0,waste_w):.1f}" height="{h}" rx="4"/>')
    labs = ""
    if use_w > 60:
        labs += f'<text class="seg-t" x="{x0+8}" y="{y+22}">useful  {g*100:.0f}%</text>'
    if waste_w > 60:
        labs += f'<text class="seg-t" x="{ux+8:.1f}" y="{y+22}">tail wait  {(1-g)*100:.0f}%</text>'
    ticks = (f'<text class="tick" x="{x0}" y="{y-6}">0s</text>'
             f'<text class="tick" x="{x1}" y="{y-6}" text-anchor="end">{_fmt_s(mk)} (makespan)</text>')
    svg = f'<svg viewBox="0 0 {W} {H}">{ticks}{seg}{labs}</svg>'
    leg = ('<div class="legend"><span><i style="background:var(--series)"></i>useful (mean completion)</span>'
           '<span><i style="background:var(--waste)"></i>tail wait (makespan − mean)</span></div>')
    return _panel("Goodput decomposition", "share of the batch wall time that is useful vs spent idle waiting on stragglers", svg + leg)


# ---------- panel 5: completion vs OSL scatter ----------
def _scatter(rep):
    rs = rep.get("rollouts") or []
    if not rs:
        return ""
    W, H, mL, mR, mT, mB = 940, 300, 52, 14, 12, 30
    x0, x1, y0, y1 = mL, W - mR, mT, H - mB
    xmax = max((r["total_osl"] for r in rs), default=1) or 1
    ymax = max((r["completion_s"] for r in rs), default=1) or 1
    sx = lambda v: x0 + (v / xmax) * (x1 - x0)
    sy = lambda v: y1 - (v / ymax) * (y1 - y0)
    g = []
    for f in (0, .25, .5, .75, 1.0):
        yv = sy(ymax * f)
        g.append(f'<line class="grid-line" x1="{x0}" y1="{yv:.1f}" x2="{x1}" y2="{yv:.1f}"/>')
        g.append(f'<text class="tick" x="{x0-6}" y="{yv+3:.1f}" text-anchor="end">{_fmt_s(ymax*f)}</text>')
    for k in range(0, 6):
        v = xmax * k / 5
        g.append(f'<text class="tick" x="{sx(v):.1f}" y="{y1+14}" text-anchor="middle">{_fmt_tok(v)}</text>')
    dots = "".join(f'<circle class="dot" cx="{sx(r["total_osl"]):.1f}" cy="{sy(r["completion_s"]):.1f}" r="3.5" '
                   f'data-osl="{r["total_osl"]}" data-comp="{r["completion_s"]:.1f}" data-turns="{r["turns"]}"/>'
                   for r in rs)
    body = ("".join(g) + _axes(x0, y0, x1, y1) + dots +
            f'<text class="axlabel" x="{(x0+x1)/2:.0f}" y="{H-1}" text-anchor="middle">rollout total output tokens</text>'
            f'<text class="axlabel" transform="translate(12,{(y0+y1)/2:.0f}) rotate(-90)" text-anchor="middle">completion (s)</text>')
    svg = f'<svg viewBox="0 0 {W} {H}" class="scatter">{body}</svg>'
    return _panel("Completion vs. rollout output length", "each dot is a rollout — a near-monotone rise confirms the tail is generation-bound, not noise", svg)


# ---------- panel 6: token-domain validation ----------
def _tokenval(rep):
    vt = rep.get("validate_token")
    if not vt:
        return ""
    realized, checks = vt.get("realized", {}), vt.get("checks", {})
    targets = {50: 654, 95: 33212, 99: 57067}   # default anchors (shown for reference)
    ps = [50, 95, 99]
    xmax = max(list(realized.values()) + list(targets.values()) + [1])
    W, H, mL, mR = 940, 150, 60, 14
    x0, x1 = mL, W - mR
    sx = lambda v: x0 + (v / xmax) * (x1 - x0)
    rows = []
    for i, p in enumerate(ps):
        y = 24 + i * 38
        r = realized.get(p) or realized.get(str(p))
        t = targets[p]
        ok = checks.get(p, checks.get(str(p)))
        col = "var(--good)" if ok else "var(--serious)"
        rows.append(f'<text class="axlabel" x="{x0-8}" y="{y+4}" text-anchor="end">p{p}</text>')
        rows.append(f'<line class="grid-line" x1="{x0}" y1="{y}" x2="{x1}" y2="{y}"/>')
        rows.append(f'<line x1="{sx(t):.1f}" y1="{y-9}" x2="{sx(t):.1f}" y2="{y+9}" stroke="var(--muted)" stroke-width="2"/>')
        rows.append(f'<text class="tick" x="{sx(t):.1f}" y="{y-13}" text-anchor="middle">target {_fmt_tok(t)}</text>')
        if r is not None:
            rows.append(f'<circle cx="{sx(r):.1f}" cy="{y}" r="6" fill="{col}" stroke="var(--panel)" stroke-width="1.5"/>')
            rows.append(f'<text class="tick" x="{sx(r)+10:.1f}" y="{y+4:.1f}">{_fmt_tok(r)}</text>')
    svg = f'<svg viewBox="0 0 {W} {H}">{"".join(rows)}</svg>'
    leg = ('<div class="legend"><span><i style="background:var(--muted)"></i>distribution target</span>'
           '<span><i style="background:var(--good)"></i>realized (within tol)</span>'
           '<span><i style="background:var(--serious)"></i>realized (off)</span></div>')
    return _panel("Token-domain validation", "realized per-rollout total OSL vs the calibrated distribution at p50 / p95 / p99", svg + leg)


def _panel(title, hint, inner):
    return f'<div class="panel"><h3>{_e(title)}</h3><p class="h">{_e(hint)}</p>{inner}</div>'


# ---------- table view ----------
def _table(rep):
    keys = ["num_sessions", "num_requests", "makespan_s", "completion_p50_s",
            "completion_p90_s", "completion_p99_s", "tail_bubble_s", "goodput_proxy",
            "output_tok_throughput", "request_throughput"]
    rows = "".join(f'<tr><td>{_e(k)}</td><td class="n">{_e(round(rep[k],4) if isinstance(rep.get(k),float) else rep.get(k))}</td></tr>'
                   for k in keys if k in rep)
    ap = rep.get("aiperf") or {}
    aprows = "".join(f'<tr><td>aiperf.{_e(k)}</td><td class="n">{_e(v)}</td></tr>'
                     for k, v in ap.items() if not isinstance(v, dict))
    return f'<div id="tableview"><table><tr><th>metric</th><th style="text-align:right">value</th></tr>{rows}{aprows}</table></div>'


_JS = """
const tip=document.getElementById('tip');
function showTip(x,y,htmlStr){tip.innerHTML=htmlStr;tip.style.left=(x+12)+'px';tip.style.top=(y+12)+'px';tip.style.opacity=1;}
function hideTip(){tip.style.opacity=0;}
document.querySelectorAll('.scatter .dot').forEach(d=>{
  d.addEventListener('mousemove',e=>showTip(e.clientX,e.clientY,
    '<b>'+(+d.dataset.comp).toFixed(1)+'s</b><br>'+d.dataset.osl+' tokens · '+d.dataset.turns+' turns'));
  d.addEventListener('mouseleave',hideTip);
});
// CDF crosshair: nearest completion by x
(function(){const svg=document.getElementById('cdf-svg');if(!svg)return;
  const comps=(window.__ROLLOUTS__||[]).map(r=>r.completion_s).sort((a,b)=>a-b);
  const x0=+svg.dataset.x0,x1=+svg.dataset.x1,y0=+svg.dataset.y0,y1=+svg.dataset.y1,xmax=+svg.dataset.xmax;
  const cx=document.getElementById('cdf-cx'),hp=document.getElementById('cdf-hp');
  svg.addEventListener('mousemove',e=>{const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;
    const p=pt.matrixTransform(svg.getScreenCTM().inverse());
    const t=Math.max(0,Math.min(xmax,(p.x-x0)/(x1-x0)*xmax));
    let i=0;while(i<comps.length&&comps[i]<t)i++; const frac=i/comps.length;
    const px=x0+(comps[Math.min(i,comps.length-1)]/xmax)*(x1-x0), py=y1-frac*(y1-y0);
    cx.setAttribute('x1',px);cx.setAttribute('x2',px);cx.style.opacity=1;
    hp.setAttribute('cx',px);hp.setAttribute('cy',py);hp.style.opacity=1;
    showTip(e.clientX,e.clientY,'<b>'+(frac*100).toFixed(0)+'%</b> of rollouts done by '+(comps[Math.min(i,comps.length-1)]).toFixed(1)+'s');});
  svg.addEventListener('mouseleave',()=>{cx.style.opacity=0;hp.style.opacity=0;hideTip();});})();
// dark mode + table toggles
const root=document.documentElement;
document.getElementById('themebtn').onclick=()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';};
document.getElementById('tablebtn').onclick=()=>{document.body.classList.toggle('showtable');};
"""


def render_report(rep, title="rl-traces-bench report"):
    """Return a self-contained HTML string for a single analyze `report` dict."""
    rollouts_json = json.dumps(rep.get("rollouts") or [])
    charts = "".join([_cdf(rep), _goodput(rep), _scatter(rep), _tokenval(rep)])
    return f"""<!doctype html><html lang="en" data-theme="light"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{_CSS}</style></head><body>
<main>
<div class="top"><div><h1>{_e(title)}</h1>
<p class="sub">static-batch long-tail replay — {_e(rep.get('num_sessions','?'))} rollouts, {_e(rep.get('num_requests','?'))} requests</p></div>
<div><button class="toggle" id="tablebtn">Table</button>
<button class="toggle" id="themebtn">Dark</button></div></div>
{_chips(rep)}
<h2>Headline</h2>{_tiles(rep)}
<div id="charts"><h2>Long tail</h2>{charts}</div>
{_table(rep)}
</main>
<div id="tip"></div>
<script>window.__ROLLOUTS__={rollouts_json};{_JS}</script>
</body></html>"""
