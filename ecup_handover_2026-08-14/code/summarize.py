"""Collect scoreboard.jsonl into a ranked summary table (CSV + markdown + JSON for the report)."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "model_out")
rows = [json.loads(l) for l in (OUT / "scoreboard.jsonl").read_text().splitlines() if l.strip()]
seen = {}
for r in rows:
    seen[r["model"]] = r
rows = sorted(seen.values(), key=lambda r: r["rmsle_val"])

NAIVE = 2.19506
for i, r in enumerate(rows, 1):
    r["rank"] = i
    r["delta_naive"] = round(r["rmsle_val"] - NAIVE, 5)
    r["gap_hon"] = None if r["rmsle_hon"] is None else round(r["rmsle_hon"] - r["rmsle_val"], 5)

(OUT / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

hdr = ["rank", "model", "family", "rmsle_val", "rmsle_hon", "delta_naive", "secs", "notes"]
csv = [",".join(hdr)]
for r in rows:
    csv.append(",".join('"' + str(r.get(h, "")).replace('"', "'") + '"' for h in hdr))
(OUT / "summary.csv").write_text("\n".join(csv))

fam_order, by_fam = [], {}
for r in rows:
    by_fam.setdefault(r["family"], []).append(r)
    if r["family"] not in fam_order: fam_order.append(r["family"])

md = ["| # | Модель | Семейство | RMSLE val | RMSLE honest | Δ к наивному |",
      "|---|---|---|---|---|---|"]
for r in rows:
    h = "—" if r["rmsle_hon"] is None else f"{r['rmsle_hon']:.4f}"
    md.append(f"| {r['rank']} | {r['model']} | {r['family']} | **{r['rmsle_val']:.4f}** | {h} | {r['delta_naive']:+.3f} |")
(OUT / "summary.md").write_text("\n".join(md))

print(f"{len(rows)} models\n")
print(f"{'#':>3} {'RMSLE':>8} {'honest':>8} {'Δnaive':>8}  model")
for r in rows:
    h = "" if r["rmsle_hon"] is None else f"{r['rmsle_hon']:8.4f}"
    print(f"{r['rank']:>3} {r['rmsle_val']:8.4f} {h:>8} {r['delta_naive']:+8.3f}  {r['model'][:64]}")
print("\nbest per family:")
for f in sorted(by_fam, key=lambda f: min(x["rmsle_val"] for x in by_fam[f])):
    b = min(by_fam[f], key=lambda x: x["rmsle_val"])
    print(f"  {f:<14s} {b['rmsle_val']:.4f}  {b['model'][:58]}  (n={len(by_fam[f])})")
