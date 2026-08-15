"""Generate the live score dashboard from results/ — one self-contained HTML file.

Contract for the autonomous agent: write results/status.json, then run this script, then
republish dashboard.html to the same artifact URL. The page reloads itself every 60s, so the
viewer sees the new state without touching anything.

    results/leaderboard.csv   #,file,what,LB           submission history
    results/summary.csv       rank,model,family,...    the 79-model battery
    results/status.json       {updated, phase, running[], log[]}   optional, written by the agent

Design tokens are lifted from the project's existing EDA / results / catalog reports so the
dashboard reads as part of the same family.
"""
import csv
import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DST = ROOT / "dashboard.html"

NAIVE = 2.19506
VAR_L = 5.366802          # Var(log1p y) on the public split, solved from the slope probe


# ------------------------------------------------------------------ data
def read_leaderboard():
    rows = []
    with open(RES / "leaderboard.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lb = float(r["LB"])
            except (TypeError, ValueError):
                continue
            what = (r.get("what") or "").strip()
            rows.append({"n": (r.get("#") or "").strip(), "file": (r.get("file") or "").strip(),
                         "what": what, "lb": lb, "probe": "probe" in what.lower()})
    return rows


def read_models(limit=15):
    p = RES / "summary.csv"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.append({"rank": int(r["rank"]), "model": r["model"], "family": r["family"],
                            "val": float(r["rmsle_val"]),
                            "hon": float(r["rmsle_hon"]) if r.get("rmsle_hon") else None})
            except (TypeError, ValueError, KeyError):
                continue
    out.sort(key=lambda d: d["rank"])
    return out[:limit]


def read_experiments(limit=40):
    p = RES / "experiments.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out.sort(key=lambda d: str(d.get("ts", "")), reverse=True)
    return out[:limit]


def read_status():
    p = RES / "status.json"
    if not p.exists():
        return {"updated": None, "phase": "нет данных", "running": [], "log": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"updated": None, "phase": "status.json не разбирается", "running": [], "log": []}


# ------------------------------------------------------------------ chart
def chart_svg(rows, w=880, h=300):
    """Submission-by-submission score, with the running best as a descending staircase.

    Normal y orientation: falling line = improving, which is how an error metric reads.
    Probes (deliberately off-optimum submissions) are drawn hollow — they are measurements,
    not attempts, and averaging them into the eye's trend line would misrepresent progress.
    """
    if not rows:
        return '<p class="empty">нет замеров</p>'
    pad_l, pad_r, pad_t, pad_b = 62, 92, 20, 34
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    vals = [r["lb"] for r in rows]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    lo, hi = lo - span * 0.18, hi + span * 0.12
    n = len(rows)

    def X(i):
        return pad_l + (iw * i / max(n - 1, 1))

    def Y(v):
        return pad_t + ih * (v - lo) / (hi - lo)

    best, best_pts = float("inf"), []
    for i, r in enumerate(rows):
        best = min(best, r["lb"])
        best_pts.append((X(i), Y(best), best))

    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="RMSLE по сабмитам, {n} замеров, лучший {min(vals):.5f}">']
    # grid + y axis
    steps = 4
    for k in range(steps + 1):
        v = lo + (hi - lo) * k / steps
        y = Y(v)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" class="grid"/>')
        o.append(f'<text x="{pad_l-10}" y="{y+3.5:.1f}" class="ax ar">{ru(v, 4)}</text>')
    # running-best staircase
    d = []
    for i, (x, y, _) in enumerate(best_pts):
        if i == 0:
            d.append(f"M{x:.1f},{y:.1f}")
        else:
            d.append(f"L{x:.1f},{best_pts[i-1][1]:.1f} L{x:.1f},{y:.1f}")
    o.append(f'<path d="{" ".join(d)}" class="beststep"/>')
    # per-submission marks
    for i, r in enumerate(rows):
        x, y = X(i), Y(r["lb"])
        cls = "dot probe" if r["probe"] else "dot"
        tip = f'{r["file"]} — {ru(r["lb"])}'
        if r["probe"]:
            tip += " · зонд"
        o.append(f'<g class="pt"><circle cx="{x:.1f}" cy="{y:.1f}" r="14" class="hit"/>'
                 f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="{cls}"/>'
                 f'<title>{html.escape(tip)}</title></g>')
        o.append(f'<text x="{x:.1f}" y="{h-pad_b+16:.0f}" class="ax am">{i+1}</text>')
    # direct label on the current best (relief for the low-contrast aqua)
    bx, by, bv = best_pts[-1]
    o.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{bx+16:.1f}" y2="{by:.1f}" class="lead"/>')
    o.append(f'<text x="{bx+21:.1f}" y="{by-4:.1f}" class="blab">{ru(bv)}</text>')
    o.append(f'<text x="{bx+21:.1f}" y="{by+10:.1f}" class="bsub">лучший</text>')
    o.append(f'<text x="{pad_l}" y="{h-4}" class="ax">№ сабмита · RMSLE, меньше — лучше</text>')
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------ page
CSS = """
:root{color-scheme:light;
 --ground:#f5f7fa;--surface:#ffffff;--surface-2:#eef2f7;--surface-3:#e6ebf2;
 --ink:#0d1117;--ink-2:#47525f;--ink-3:#79848f;
 --rule:#e0e6ee;--rule-strong:#ccd5e0;
 --accent:#2a78d6;--accent-soft:#e8f0fc;
 --alert:#eb6834;--alert-soft:#fdeee7;
 --aqua:#1baf7a;--aqua-soft:#e5f6ef;
 --shadow:0 1px 2px rgba(13,17,23,.05),0 8px 24px -16px rgba(13,17,23,.22);}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --ground:#0c1015;--surface:#141a22;--surface-2:#1b222c;--surface-3:#232b36;
 --ink:#e9eef4;--ink-2:#a0abb9;--ink-3:#6e7a88;
 --rule:#232b36;--rule-strong:#323c49;
 --accent:#3987e5;--accent-soft:#16283f;
 --alert:#d95926;--alert-soft:#331c12;
 --aqua:#22c98d;--aqua-soft:#0f2b22;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8);}}
:root[data-theme="dark"]{color-scheme:dark;
 --ground:#0c1015;--surface:#141a22;--surface-2:#1b222c;--surface-3:#232b36;
 --ink:#e9eef4;--ink-2:#a0abb9;--ink-3:#6e7a88;
 --rule:#232b36;--rule-strong:#323c49;
 --accent:#3987e5;--accent-soft:#16283f;
 --alert:#d95926;--alert-soft:#331c12;
 --aqua:#22c98d;--aqua-soft:#0f2b22;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -18px rgba(0,0,0,.8);}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
 font-size:15.5px;line-height:1.62;letter-spacing:-.003em;-webkit-font-smoothing:antialiased}
h1,h2,.eyebrow,th,.tile-v,.mono,.btn,code,.fam,.ax,.blab,.bsub{
 font-family:ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px 80px;min-width:0}
.mast{padding:40px 0 0;border-bottom:1px solid var(--rule);margin-bottom:26px}
.eyebrow{font-size:10.5px;letter-spacing:.19em;text-transform:uppercase;color:var(--accent);
 font-weight:600;margin:0 0 14px}
h1{font-size:clamp(26px,4vw,40px);line-height:1.07;letter-spacing:-.035em;font-weight:680;
 margin:0 0 16px;max-width:22ch;text-wrap:balance}
.deck{font-size:16.5px;line-height:1.55;color:var(--ink-2);max-width:64ch;margin:0 0 22px}
.deck b{color:var(--ink);font-weight:600}
section{margin:0 0 42px}
.sec-head{display:flex;align-items:baseline;gap:12px;margin:0 0 14px;padding-top:6px;
 border-top:2px solid var(--ink)}
h2{font-size:21px;line-height:1.2;letter-spacing:-.03em;font-weight:660;margin:0}
.sec-note{font-size:12px;color:var(--ink-3);margin-left:auto;font-family:ui-monospace,Menlo,monospace}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1px;
 background:var(--rule);border:1px solid var(--rule);border-radius:10px;overflow:hidden;margin:0 0 34px}
.tile{background:var(--surface);padding:15px 17px 14px}
.tile-k{font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);margin:0 0 7px;
 font-family:ui-monospace,Menlo,monospace}
.tile-v{font-size:25px;line-height:1;font-weight:640;letter-spacing:-.028em;margin:0 0 5px;
 font-variant-numeric:tabular-nums}
.tile-v.good{color:var(--aqua)}
.tile-v.warn{color:var(--alert)}
.tile-s{font-size:11.5px;color:var(--ink-3);line-height:1.35;margin:0}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
 box-shadow:var(--shadow);padding:16px 18px}
.grid{stroke:var(--rule);stroke-width:1}
.ax{font-size:10.5px;fill:var(--ink-3)}
.ar{text-anchor:end}.am{text-anchor:middle}
.beststep{fill:none;stroke:var(--aqua);stroke-width:2;stroke-linejoin:round}
.dot{fill:var(--accent);stroke:var(--surface);stroke-width:2}
.dot.probe{fill:var(--surface);stroke:var(--ink-3);stroke-width:2}
.hit{fill:transparent}
.pt:hover .dot{stroke:var(--ink)}
.lead{stroke:var(--aqua);stroke-width:1.5}
.blab{font-size:13px;font-weight:660;fill:var(--aqua);font-variant-numeric:tabular-nums}
.bsub{font-size:10px;fill:var(--ink-3);letter-spacing:.08em;text-transform:uppercase}
.key{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;font-size:12px;color:var(--ink-2)}
.key i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}
.key .k1 i{background:var(--accent)}
.key .k2 i{background:transparent;box-shadow:inset 0 0 0 2px var(--ink-3)}
.key .k3 i{background:var(--aqua);border-radius:1px;height:3px;width:16px}
.tscroll{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--surface);
 box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-3);font-weight:600;
 text-align:right;padding:10px 12px;border-bottom:1px solid var(--rule-strong);background:var(--surface-2);
 white-space:nowrap}
th:nth-child(2),th:nth-child(3){text-align:left}
td{padding:9px 12px;border-bottom:1px solid var(--rule);color:var(--ink-2);text-align:right;white-space:nowrap}
td.nm{color:var(--ink);font-weight:600;text-align:left;white-space:normal;min-width:190px}
td.no{text-align:left}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--surface-2)}
tr.best td{background:var(--aqua-soft)}
tr.dim td{color:var(--ink-3)}
.fam{font-size:10px;padding:2.5px 7px;border-radius:4px;background:var(--surface-3);color:var(--ink-2);
 white-space:nowrap;font-family:ui-monospace,Menlo,monospace}
.chip{font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;font-weight:600;padding:2.5px 7px;
 border-radius:4px;box-shadow:inset 0 0 0 1px currentColor;font-family:ui-monospace,Menlo,monospace}
.chip.run{color:var(--accent);background:var(--accent-soft)}
.chip.idle{color:var(--ink-3);background:var(--surface-3)}
.chip.closed{color:var(--ink-3);background:var(--surface-3)}
.chip.surv{color:var(--aqua);background:var(--aqua-soft)}
td.good{color:var(--aqua);font-weight:640}
td.bad{color:var(--ink-3)}
td.note{text-align:left;white-space:normal;font-size:12px;color:var(--ink-2);
 max-width:330px;min-width:210px}
td.nm .sub,td.note .sub{display:block;font-weight:400;font-size:11.5px;color:var(--ink-3);
 margin-top:3px;white-space:normal}
td.mono{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--ink-3)}
.runlist{list-style:none;margin:0;padding:0;display:grid;gap:1px;background:var(--rule);
 border:1px solid var(--rule);border-radius:10px;overflow:hidden}
.runlist li{background:var(--surface);padding:13px 16px;display:flex;gap:12px;align-items:baseline;
 flex-wrap:wrap}
.runlist .nm{font-weight:640;font-family:ui-monospace,Menlo,monospace;font-size:13px}
.runlist .meta{color:var(--ink-3);font-size:12px;margin-left:auto;font-family:ui-monospace,Menlo,monospace}
.runlist .note{flex-basis:100%;color:var(--ink-2);font-size:13px;margin:0}
.empty{color:var(--ink-3);font-size:13px;margin:0}
.qbox{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:10px;
 overflow:hidden;box-shadow:var(--shadow)}
.q{background:var(--surface);padding:15px 17px;display:grid;grid-template-columns:30px 1fr;gap:13px}
.q.open{background:var(--alert-soft)}
.qn{font-family:ui-monospace,Menlo,monospace;font-size:12px;font-weight:660;color:var(--alert);
 padding-top:2px}
.q.done .qn{color:var(--ink-3)}
.qt{margin:0;font-weight:600;color:var(--ink)}
.q.done .qt{font-weight:400;color:var(--ink-2)}
.qc{margin:5px 0 0;font-size:13px;color:var(--ink-2)}
.qo{margin:9px 0 0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:7px}
.qo li{font-size:12px;padding:3px 9px;border-radius:5px;background:var(--surface-2);
 border:1px solid var(--rule-strong);font-family:ui-monospace,Menlo,monospace}
.qa{margin:8px 0 0;font-size:13px;color:var(--aqua);font-weight:600}
.howto{font-size:12.5px;color:var(--ink-3);margin:11px 0 0}
.queue{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:10px;
 overflow:hidden}
.qi{background:var(--surface);padding:14px 17px;display:grid;grid-template-columns:26px 1fr auto;
 gap:13px;align-items:baseline}
.qi .sl{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--ink-3)}
.qi .fn{font-family:ui-monospace,Menlo,monospace;font-size:13px;font-weight:640;color:var(--ink)}
.qi .why{grid-column:2;font-size:13px;color:var(--ink-2);margin:4px 0 0}
.qi .exp{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:var(--accent);
 font-variant-numeric:tabular-nums;white-space:nowrap}
.logl{list-style:none;margin:0;padding:0;font-size:13px}
.logl li{display:flex;gap:14px;padding:7px 0;border-bottom:1px solid var(--rule)}
.logl li:last-child{border-bottom:none}
.logl time{color:var(--ink-3);font-family:ui-monospace,Menlo,monospace;font-size:11.5px;
 white-space:nowrap;padding-top:2px}
footer{border-top:1px solid var(--rule);margin-top:14px;padding-top:18px;font-size:12px;color:var(--ink-3);
 font-family:ui-monospace,Menlo,monospace;display:flex;flex-wrap:wrap;gap:6px 22px;align-items:center}
.btn{font-size:11.5px;padding:4px 10px;border-radius:6px;border:1px solid var(--rule-strong);
 background:var(--surface-2);color:var(--ink-2);cursor:pointer}
.btn:hover{color:var(--ink);border-color:var(--accent)}
.btn[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
(function(){
 var on=true,left=60,el=document.getElementById('cd'),b=document.getElementById('pz');
 function tick(){ if(!on){return;} left--; el.textContent=left+' с';
  if(left<=0){location.reload();} }
 setInterval(tick,1000);
 b.addEventListener('click',function(){on=!on;b.setAttribute('aria-pressed',String(!on));
  b.textContent=on?'пауза':'продолжить'; el.textContent=on?left+' с':'на паузе';});
})();
"""


def ru(x, nd=5):
    return f"{x:.{nd}f}".replace(".", ",")


def build():
    lb = read_leaderboard()
    models = read_models()
    st = read_status()
    best = min((r["lb"] for r in lb), default=None)
    real = [r for r in lb if not r["probe"]]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    tiles = []
    if best is not None:
        tiles.append(("Лучший скор", ru(best), "RMSLE на публичном лидерборде", "good"))
        tiles.append(("Выигрыш у бейзлайна", ru(NAIVE - best, 3),
                      f"{ru(NAIVE)} → {ru(best)}", ""))
        expl = 1 - best ** 2 / VAR_L
        tiles.append(("Объяснено дисперсии", f"{expl*100:.1f} %".replace(".", ","),
                      f"из Var(log1p y) = {ru(VAR_L, 4)}", ""))
    tiles.append(("Сабмитов", str(len(lb)),
                  f"из них зондов: {len(lb)-len(real)}", ""))
    sub = st.get("submissions") or {}
    used, limit = int(sub.get("used", 0)), int(sub.get("limit", 3))
    tiles.append(("Попыток сегодня", f"{used} / {limit}",
                  f"осталось {max(limit-used,0)} · {sub.get('date','')}",
                  "" if used < limit else "warn"))
    tiles.append(("Моделей испытано", str(len(read_models(10**6))), "в 9 семействах", ""))

    T = []
    for k, v, s, cls in tiles:
        T.append(f'<div class="tile"><p class="tile-k">{k}</p>'
                 f'<p class="tile-v {cls}">{v}</p><p class="tile-s">{html.escape(s)}</p></div>')

    # submissions table
    run_best, S = float("inf"), []
    for i, r in enumerate(lb):
        is_new = r["lb"] < run_best
        run_best = min(run_best, r["lb"])
        cls = "best" if r["lb"] == best else ("dim" if r["probe"] else "")
        mark = "&darr;" if is_new and not r["probe"] else ""
        S.append(f'<tr class="{cls}"><td class="no">{i+1}</td>'
                 f'<td class="nm">{html.escape(r["file"])}</td>'
                 f'<td class="no">{html.escape(r["what"])}</td>'
                 f'<td><b>{ru(r["lb"])}</b> {mark}</td></tr>')

    # model battery
    M = []
    for m in models:
        hon = ru(m["hon"], 4) if m["hon"] is not None else "—"
        M.append(f'<tr><td class="no">{m["rank"]}</td><td class="nm">{html.escape(m["model"])}</td>'
                 f'<td class="no"><span class="fam">{html.escape(m["family"])}</span></td>'
                 f'<td><b>{ru(m["val"], 4)}</b></td><td>{hon}</td></tr>')

    # status
    running = st.get("running") or []
    if running:
        R = "".join(
            f'<li><span class="chip run">идёт</span><span class="nm">{html.escape(str(r.get("name","")))}</span>'
            f'<span class="meta">{html.escape(str(r.get("started","")))}</span>'
            + (f'<p class="note">{html.escape(str(r["note"]))}</p>' if r.get("note") else "")
            + "</li>" for r in running)
        R = f'<ul class="runlist">{R}</ul>'
    else:
        R = ('<ul class="runlist"><li><span class="chip idle">простой</span>'
             '<span class="nm">ничего не считается</span></li></ul>')

    log = st.get("log") or []
    if log:
        L = "".join(f'<li><time>{html.escape(str(e.get("ts","")))}</time>'
                    f'<span>{html.escape(str(e.get("event","")))}</span></li>' for e in log[:14])
        L = f'<ul class="logl">{L}</ul>'
    else:
        L = '<p class="empty">журнал пуст — агент ещё ничего не записал</p>'

    # open questions first — they are the only thing on the page that needs the reader to act
    qs = st.get("questions") or []
    qs = sorted(qs, key=lambda q: (q.get("status") == "answered", -int(q.get("id", 0) or 0)))
    n_open = sum(1 for q in qs if q.get("status") != "answered")
    if qs:
        Q = []
        for q in qs:
            done = q.get("status") == "answered"
            opts = "".join(f"<li>{html.escape(str(o))}</li>" for o in (q.get("options") or []))
            Q.append(
                f'<div class="q {"done" if done else "open"}"><span class="qn">#{html.escape(str(q.get("id","?")))}</span>'
                f'<div><p class="qt">{html.escape(str(q.get("text","")))}</p>'
                + (f'<p class="qc">{html.escape(str(q["context"]))}</p>' if q.get("context") else "")
                + (f'<ul class="qo">{opts}</ul>' if opts else "")
                + (f'<p class="qa">Ответ: {html.escape(str(q.get("answer","")))}</p>' if done else "")
                + "</div></div>")
        Q = (f'<div class="qbox">{"".join(Q)}</div>'
             '<p class="howto">Ответить: написать агенту в чат, сославшись на номер — '
             '«по вопросу 3: делай так». Срочности нет, агент продолжает работать без ответа.</p>')
    else:
        Q = '<div class="qbox"><div class="q done"><span class="qn">—</span>' \
            '<div><p class="qt">Вопросов нет</p></div></div></div>'

    # submission queue
    queue = st.get("queue") or []
    if queue:
        QU = "".join(
            f'<div class="qi"><span class="sl">{i+1}</span>'
            f'<span class="fn">{html.escape(str(q.get("file","")))}</span>'
            f'<span class="exp">{html.escape(str(q.get("expected","")))}</span>'
            f'<p class="why">{html.escape(str(q.get("why","")))}</p></div>' for i, q in enumerate(queue))
        QU = f'<div class="queue">{QU}</div>'
    else:
        QU = '<div class="queue"><div class="qi"><span class="sl">—</span>' \
             '<span class="fn">очередь пуста</span><span class="exp"></span>' \
             '<p class="why">Отправлять пока нечего: кандидатов, которые бьют текущий стек, нет.</p></div></div>'

    # experiment ledger — every hypothesis with its number and how it died
    exps = read_experiments()
    n_closed = sum(1 for e in exps if e.get("verdict") == "closed")
    E = []
    for e in exps:
        v = e.get("verdict", "")
        chip = ('<span class="chip closed">закрыто</span>' if v == "closed"
                else '<span class="chip surv">выжило</span>' if v == "survived"
                else '<span class="chip idle">—</span>')
        try:
            d = float(e["rmsle"]) - float(e["baseline"])
            delta = f'{ru(d, 5)}' if d >= 0 else f'{ru(d, 5)}'
            dcls = "good" if d < 0 else "bad"
        except (KeyError, TypeError, ValueError):
            delta, dcls = "—", ""
        E.append(
            f'<tr><td class="no mono">{html.escape(str(e.get("ts","")))[-5:]}</td>'
            f'<td class="nm">{html.escape(str(e.get("name","")))}'
            f'<span class="sub">{html.escape(str(e.get("hypothesis","")))}</span></td>'
            f'<td class="no"><span class="fam">{html.escape(str(e.get("protocol","")))}</span></td>'
            f'<td>{ru(float(e["rmsle"]), 5) if e.get("rmsle") is not None else "—"}</td>'
            f'<td class="{dcls}">{delta}</td>'
            f'<td class="no">{chip}</td>'
            f'<td class="note">{html.escape(str(e.get("control","")))}<span class="sub">'
            f'{html.escape(str(e.get("note","")))}</span></td></tr>')

    phase = html.escape(str(st.get("phase") or "—"))
    updated = html.escape(str(st.get("updated") or now))

    page = f"""<title>Пульт скора E-CUP</title>
<style>{CSS}</style>
<div class="wrap">
<header class="mast">
  <p class="eyebrow">Ozon E-CUP · Search LTV · пульт</p>
  <h1>Скор в реальном времени</h1>
  <p class="deck">Публичный лидерборд, батарея моделей и то, что считается прямо сейчас.
  Страница пересобирается из <code>results/</code> и обновляется сама — держать открытой
  и не трогать. Текущая фаза: <b>{phase}</b>.</p>
</header>

<div class="tiles">{"".join(T)}</div>

<section>
  <div class="sec-head"><h2>Вопросы агента</h2>
    <span class="sec-note">{n_open} без ответа</span></div>
  {Q}
</section>

<section>
  <div class="sec-head"><h2>Очередь на отправку</h2>
    <span class="sec-note">лимит {limit} в день</span></div>
  {QU}
</section>

<section>
  <div class="sec-head"><h2>Динамика скора</h2>
    <span class="sec-note">{len(lb)} замеров</span></div>
  <div class="card">{chart_svg(lb)}
    <div class="key">
      <span class="k1"><i></i>сабмит</span>
      <span class="k2"><i></i>зонд — намеренно неоптимальный, замер ради алгебры</span>
      <span class="k3"><i></i>лучший на текущий момент</span>
    </div>
  </div>
</section>

<section>
  <div class="sec-head"><h2>Что считается сейчас</h2>
    <span class="sec-note">обновлено {updated}</span></div>
  {R}
</section>

<section>
  <div class="sec-head"><h2>Журнал</h2></div>
  <div class="card">{L}</div>
</section>

<section>
  <div class="sec-head"><h2>Журнал экспериментов</h2>
    <span class="sec-note">{len(exps)} проверено · {n_closed} закрыто</span></div>
  <div class="tscroll"><table>
    <thead><tr><th>время</th><th>эксперимент и гипотеза</th><th>протокол</th><th>RMSLE</th>
      <th>к базе</th><th>итог</th><th>контроль и что вышло</th></tr></thead>
    <tbody>{"".join(E)}</tbody></table></div>
</section>

<section>
  <div class="sec-head"><h2>Сабмиты</h2>
    <span class="sec-note">&darr; — новый лучший</span></div>
  <div class="tscroll"><table>
    <thead><tr><th>#</th><th>Файл</th><th>Что это</th><th>LB</th></tr></thead>
    <tbody>{"".join(S)}</tbody></table></div>
</section>

<section>
  <div class="sec-head"><h2>Топ моделей на валидации</h2>
    <span class="sec-note">якорь 2026-01-14 · honest 2025-10-16</span></div>
  <div class="tscroll"><table>
    <thead><tr><th>#</th><th>Модель</th><th>Семейство</th><th>RMSLE val</th><th>RMSLE honest</th></tr></thead>
    <tbody>{"".join(M)}</tbody></table></div>
</section>

<footer>
  <span>Ozon E-CUP · Search LTV</span>
  <span>собрано {now}</span>
  <span>обновление через <b id="cd">60 с</b></span>
  <button class="btn" id="pz" type="button" aria-pressed="false">пауза</button>
</footer>
</div>
<script>{JS}</script>
"""
    DST.write_text(page, encoding="utf-8")
    print(f"wrote {DST}  ({DST.stat().st_size//1024} KB)")
    print(f"  submissions {len(lb)}, best {ru(best) if best else '—'}, models listed {len(models)}")
    return DST


if __name__ == "__main__":
    build()
