"""
Animated pieces of the app, as small self-contained HTML/JavaScript pages (no internet needed): count-up cards, the
possible futures of one shelf, and the replay of the whole shop day by day.
"""

from __future__ import annotations

import html as _html
import json

import numpy as np


def _js(obj):
    """JSON for a <script> block (text such as product names cannot close the script)."""
    return json.dumps(obj).replace("</", "<\\/")


def _e(text):
    return _html.escape(str(text), quote=True)

LIGHT = dict(bg="#ffffff", panel="#f6f7f9", ink="#1f2328", muted="#6b7280", grid="#e5e7eb", accent="#2a78d6",
             ok="#1baf7a", low="#eda100", out="#e34948", over="#6250d6", path="#2a78d6", path2="#9aa3ae")
DARK = dict(bg="#0e1117", panel="#161b22", ink="#e6edf3", muted="#9aa4b2", grid="#2d333b", accent="#4c9be8",
            ok="#2fbf8a", low="#e3a21a", out="#f0645f", over="#9085e9", path="#58a6ff", path2="#6e7681")
FONT = "'Source Sans Pro','Segoe UI',Helvetica,Arial,sans-serif"


def _pal(dark):
    return DARK if dark else LIGHT


def kpi_cards(cards, dark=False):
    """cards: [(label, value, prefix, suffix, decimals, note, colour key)] -> HTML with numbers counting up."""
    c = _pal(dark)
    items = []
    for k, (label, value, prefix, suffix, dec, note, col) in enumerate(cards):
        items.append(f"""<div class="card" style="animation-delay:{k * 0.08:.2f}s">
  <div class="lab">{_e(label)}</div>
  <div class="val" style="color:{c.get(col, col)}"><span>{_e(prefix)}</span><span class="num" data-v="{float(value)}"
     data-d="{int(dec)}">0</span><span class="suf">{_e(suffix)}</span></div>
  <div class="note">{_e(note)}</div></div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:transparent;font-family:{FONT};color:{c['ink']}}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.card{{background:{c['panel']};border:1px solid {c['grid']};border-radius:12px;padding:14px 16px;
  animation:rise .6s ease-out both}}
.lab{{font-size:12.5px;color:{c['muted']};letter-spacing:.3px}}
.val{{font-size:30px;font-weight:700;margin:4px 0 2px;font-variant-numeric:tabular-nums}}
.suf{{font-size:16px;font-weight:600;margin-left:2px}}
.note{{font-size:12.5px;color:{c['muted']}}}
@keyframes rise{{from{{opacity:0;transform:translateY(8px)}}}}
@media (prefers-reduced-motion:reduce){{.card{{animation:none}}}}
</style></head><body><div class="grid">{''.join(items)}</div><script>
const ease = t => 1 - Math.pow(1 - t, 3);
document.querySelectorAll('.num').forEach((el, i) => {{
  const v = parseFloat(el.dataset.v), d = parseInt(el.dataset.d), t0 = performance.now() + i * 80, dur = 1100;
  const fmt = x => x.toLocaleString('en-US', {{minimumFractionDigits: d, maximumFractionDigits: d}});
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{ el.textContent = fmt(v); return; }}
  const step = now => {{ const t = Math.min(Math.max((now - t0) / dur, 0), 1); el.textContent = fmt(v * ease(t));
    if (t < 1) requestAnimationFrame(step); }};
  requestAnimationFrame(step);
}});
</script></body></html>"""


def futures(paths_without, paths_with, lead, review, start_label, qty, unit="units", dark=False, height=360):
    """Possible futures of one shelf: stock day by day in each future, without and with today's order. The paths are
    drawn one after another; futures that empty the shelf turn red."""
    c = _pal(dark)
    data = {"a": np.asarray(paths_without).round(1).tolist(), "b": np.asarray(paths_with).round(1).tolist(),
            "lead": int(lead), "review": int(review), "qty": float(qty), "start": start_label, "unit": unit}
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:{FONT};color:{c['ink']};background:transparent}}
.bar{{display:flex;gap:8px;align-items:center;margin:2px 0 6px;flex-wrap:wrap}}
button{{font:600 13px {FONT};border:1px solid {c['grid']};background:{c['panel']};color:{c['ink']};
  border-radius:8px;padding:5px 12px;cursor:pointer}}
button.on{{background:{c['accent']};border-color:{c['accent']};color:#fff}}
.msg{{font-size:14px;margin-left:6px}} .msg b{{font-variant-numeric:tabular-nums}}
svg text{{font-family:{FONT}}}
</style></head><body>
<div class="bar"><button id="b0" class="on">Without an order</button><button id="b1">With today's order</button>
<button id="rp">↻ Replay</button><span class="msg" id="msg"></span></div>
<svg id="s" width="100%" height="{height - 40}"></svg>
<script>
const D = {_js(data)};
const C = {_js(c)};
const svg = document.getElementById('s'); const NS = 'http://www.w3.org/2000/svg';
let mode = 0, timer = null;
function el(t, a, p) {{ const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]);
  (p || svg).appendChild(e); return e; }}
function draw() {{
  if (timer) {{ clearTimeout(timer); timer = null; }}
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const P = mode ? D.b : D.a, n = P.length, days = P[0].length - 1;
  const W = svg.clientWidth || 600, H = svg.clientHeight || 300, L = 46, R = 12, T = 12, B = 30;
  const top = Math.max(1, ...D.a.flat(), ...D.b.flat()) * 1.08;
  const x = d => L + (W - L - R) * d / days, y = v => T + (H - T - B) * (1 - v / top);
  for (let k = 0; k <= 4; k++) {{ const v = top * k / 4;
    el('line', {{x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: C.grid}});
    el('text', {{x: L - 6, y: y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: C.muted}}).textContent = Math.round(v); }}
  for (let d = 0; d <= days; d += Math.max(1, Math.round(days / 8))) {{
    el('text', {{x: x(d), y: H - 10, 'text-anchor': 'middle', 'font-size': 11, fill: C.muted}}).textContent =
      d === 0 ? 'today' : '+' + d + ' d'; }}
  el('rect', {{x: x(D.lead), y: T, width: Math.max(2, x(Math.min(days, D.lead + D.review)) - x(D.lead)),
    height: H - T - B, fill: C.accent, opacity: .07}});
  el('line', {{x1: x(D.lead), x2: x(D.lead), y1: T, y2: H - B, stroke: C.accent, 'stroke-dasharray': '4 3'}});
  el('text', {{x: x(D.lead) + 4, y: T + 12, 'font-size': 11, fill: C.accent}}).textContent =
    mode ? 'order of ' + D.qty + ' ' + D.unit + ' arrives' : 'an order would arrive';
  el('line', {{x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: C.out, 'stroke-width': 1.5}});
  let empty = 0, shown = 0;
  const msg = document.getElementById('msg');
  function one(i) {{
    if (i >= n) {{ msg.innerHTML = 'The shelf runs empty in <b>' + empty + ' of ' + n +
      '</b> possible futures ' + (mode ? 'with' : 'without') + ' today\\'s order.'; return; }}
    const p = P[i]; const hit = p.slice(1).some(v => v <= 0); if (hit) empty++;
    const d = p.map((v, k) => (k ? 'L' : 'M') + x(k).toFixed(1) + ' ' + y(Math.max(v, 0)).toFixed(1)).join(' ');
    const path = el('path', {{d: d, fill: 'none', stroke: hit ? C.out : C.path, 'stroke-width': 1.4,
      opacity: hit ? .75 : .38}});
    const len = path.getTotalLength(); path.style.strokeDasharray = len; path.style.strokeDashoffset = len;
    path.style.transition = 'stroke-dashoffset .5s ease-out'; requestAnimationFrame(() => path.style.strokeDashoffset = 0);
    if (hit) {{ const k = p.findIndex((v, j) => j > 0 && v <= 0);
      el('circle', {{cx: x(k), cy: y(0), r: 3.2, fill: C.out}}); }}
    shown++; msg.innerHTML = 'Future <b>' + shown + '</b> of ' + n + ' · shelf empty in <b>' + empty + '</b>';
    timer = setTimeout(() => one(i + 1), 45);
  }}
  one(0);
}}
const b0 = document.getElementById('b0'), b1 = document.getElementById('b1');
b0.onclick = () => {{ mode = 0; b0.className = 'on'; b1.className = ''; draw(); }};
b1.onclick = () => {{ mode = 1; b1.className = 'on'; b0.className = ''; draw(); }};
document.getElementById('rp').onclick = draw;
window.addEventListener('load', draw);
</script></body></html>"""


def shelf_replay(names, status_a, status_b, lost_a, lost_b, stock_a, stock_b, dates, labels=("Your way",
                 "Stock Planner"), currency="Rs", dark=False, height=560):
    """The whole shop day by day, two ways of ordering side by side. status: (n, D) codes 0 empty, 1 low, 2 fine,
    3 too much; lost: (D,) lost sales value per day; stock: (D,) stock value at the end of each day."""
    c = _pal(dark)
    data = {"names": list(names), "a": np.asarray(status_a, int).T.tolist(), "b": np.asarray(status_b, int).T.tolist(),
            "la": np.cumsum(lost_a).round().tolist(), "lb": np.cumsum(lost_b).round().tolist(),
            "sa": np.asarray(stock_a).round().tolist(), "sb": np.asarray(stock_b).round().tolist(),
            "dates": list(dates), "labels": list(labels), "cur": currency}
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:{FONT};color:{c['ink']};background:transparent}}
.top{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}}
button{{font:600 13px {FONT};border:1px solid {c['accent']};background:{c['accent']};color:#fff;border-radius:8px;
  padding:5px 14px;cursor:pointer;min-width:84px}}
input[type=range]{{flex:1;min-width:160px;accent-color:{c['accent']}}}
.day{{font-weight:700;font-variant-numeric:tabular-nums;min-width:120px}}
.wrap{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.panel{{background:{c['panel']};border:1px solid {c['grid']};border-radius:12px;padding:10px 12px}}
.h{{font-weight:700;font-size:15px;margin-bottom:4px}}
.k{{display:flex;gap:16px;font-size:12.5px;color:{c['muted']};margin-bottom:6px;flex-wrap:wrap}}
.k b{{color:{c['ink']};font-size:15px;font-variant-numeric:tabular-nums}}
canvas{{width:100%;display:block}}
.leg{{display:flex;gap:14px;font-size:12px;color:{c['muted']};margin-top:8px;flex-wrap:wrap}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}}
@media (max-width:700px){{.wrap{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="top"><button id="pp">▶ Play</button><input id="sl" type="range" min="0" value="0">
<span class="day" id="dl"></span></div>
<div class="wrap"><div class="panel"><div class="h" id="ha"></div><div class="k" id="ka"></div>
<canvas id="ca"></canvas><canvas id="ga" height="70"></canvas></div>
<div class="panel"><div class="h" id="hb"></div><div class="k" id="kb"></div>
<canvas id="cb"></canvas><canvas id="gb" height="70"></canvas></div></div>
<div class="leg"><span><span class="sw" style="background:{c['out']}"></span>empty shelf</span>
<span><span class="sw" style="background:{c['low']}"></span>under 3 days left</span>
<span><span class="sw" style="background:{c['ok']}"></span>fine</span>
<span><span class="sw" style="background:{c['over']}"></span>over 2 months of stock</span>
<span>· one square per product · lines: lost sales so far (red) and stock value (blue)</span></div>
<script>
const D = {_js(data)}; const C = {_js(c)};
const COL = [C.out, C.low, C.ok, C.over]; const N = D.names.length, T = D.dates.length;
const sl = document.getElementById('sl'); sl.max = T - 1;
document.getElementById('ha').textContent = D.labels[0]; document.getElementById('hb').textContent = D.labels[1];
const cols = Math.ceil(Math.sqrt(N * 2.2)), rows = Math.ceil(N / cols);
function fmt(v) {{ return D.cur + ' ' + Math.round(v).toLocaleString('en-US'); }}
function setup(cv) {{ const w = cv.clientWidth || 420, s = Math.floor(w / cols); cv.width = w * devicePixelRatio;
  cv.height = (s * rows + 2) * devicePixelRatio; cv.style.height = (s * rows + 2) + 'px'; return s; }}
function grid(cv, st, day) {{ const s = setup(cv), g = cv.getContext('2d'); g.scale(devicePixelRatio, devicePixelRatio);
  for (let i = 0; i < N; i++) {{ const r = Math.floor(i / cols), q = i % cols; g.fillStyle = COL[st[day][i]];
    g.beginPath(); g.roundRect ? g.roundRect(q * s + 1, r * s + 1, s - 3, s - 3, 3) : g.rect(q * s + 1, r * s + 1, s - 3, s - 3);
    g.fill(); }} }}
function spark(cv, lost, stock, day) {{ const w = cv.clientWidth || 420, h = 70; cv.width = w * devicePixelRatio;
  cv.height = h * devicePixelRatio; cv.style.height = h + 'px'; const g = cv.getContext('2d');
  g.scale(devicePixelRatio, devicePixelRatio); const mL = Math.max(1, D.la[T - 1], D.lb[T - 1]);
  const mS = Math.max(1, ...D.sa, ...D.sb); const x = k => 4 + (w - 8) * k / (T - 1);
  g.strokeStyle = C.grid; g.beginPath(); g.moveTo(4, h - 4); g.lineTo(w - 4, h - 4); g.stroke();
  [[stock, mS, C.accent], [lost, mL, C.out]].forEach(([v, m, col]) => {{ g.strokeStyle = col; g.lineWidth = 2;
    g.beginPath(); for (let k = 0; k <= day; k++) {{ const yy = h - 6 - (h - 12) * v[k] / m;
      k ? g.lineTo(x(k), yy) : g.moveTo(x(k), yy); }} g.stroke(); }}); }}
function show(day) {{ sl.value = day; document.getElementById('dl').textContent = D.dates[day];
  grid(document.getElementById('ca'), D.a, day); grid(document.getElementById('cb'), D.b, day);
  spark(document.getElementById('ga'), D.la, D.sa, day); spark(document.getElementById('gb'), D.lb, D.sb, day);
  const e = (st) => st[day].filter(v => v === 0).length;
  document.getElementById('ka').innerHTML = 'empty today <b>' + e(D.a) + '</b> lost so far <b>' + fmt(D.la[day]) +
    '</b> stock <b>' + fmt(D.sa[day]) + '</b>';
  document.getElementById('kb').innerHTML = 'empty today <b>' + e(D.b) + '</b> lost so far <b>' + fmt(D.lb[day]) +
    '</b> stock <b>' + fmt(D.sb[day]) + '</b>'; }}
let play = null; const pp = document.getElementById('pp');
function stop() {{ clearInterval(play); play = null; pp.textContent = '▶ Play'; }}
pp.onclick = () => {{ if (play) {{ stop(); return; }} if (+sl.value >= T - 1) sl.value = 0; pp.textContent = '❚❚ Pause';
  play = setInterval(() => {{ const d = +sl.value + 1; if (d >= T) {{ stop(); return; }} show(d); }}, 140); }};
sl.oninput = () => {{ stop(); show(+sl.value); }};
window.addEventListener('resize', () => show(+sl.value));
window.addEventListener('load', () => {{ show(0); setTimeout(() => pp.click(), 600); }});
</script></body></html>"""


def status_codes(on_hand, daily, over_days=60):
    """0 empty, 1 under 3 days left, 2 fine, 3 more than over_days of stock (daily = typical daily sales)."""
    oh = np.asarray(on_hand, float)
    d = np.maximum(np.asarray(daily, float), 1e-6)[:, None]
    cover = oh / d
    return np.where(oh <= 0, 0, np.where(cover < 3, 1, np.where(cover > over_days, 3, 2)))
