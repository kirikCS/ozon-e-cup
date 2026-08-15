"""Post-run tidy-up: correct stale labels, drop orphan prediction files, rebuild the summary."""
import json
import re
from pathlib import Path

OUT = Path("model_out")
sb = OUT / "scoreboard.jsonl"
rows = [json.loads(l) for l in sb.read_text().splitlines() if l.strip()]

# the sequence window was shortened from 364 to 182 days after the padding fix, but the job
# labels still carried the old number
RENAME = {"Dilated CNN over 364 daily steps": "Dilated CNN over 182 daily steps",
          "Dilated CNN + tabular features (hybrid)": "Dilated CNN 182d + tabular features (hybrid)"}
for r in rows:
    r["model"] = RENAME.get(r["model"], r["model"])
    # the honest anchor (2025-10-16) has only 289 days of history, so a 365-day window is
    # undefined there; the number that was recorded is meaningless, not a score
    if r["model"].startswith("prev365"):
        r["rmsle_hon"] = None
        r["notes"] = (r.get("notes", "") + " | honest anchor has <365d of history, not evaluable").strip(" |")

seen = {}
for r in rows:
    seen[r["model"]] = r
rows = list(seen.values())
sb.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

# drop prediction files that no longer correspond to a scoreboard row
def key(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())
valid = {key(r["model"])[:52] for r in rows}
dropped = 0
for f in OUT.glob("pred_*.npy"):
    if not any(key(f.stem[5:])[:40] in v or v[:40] in key(f.stem[5:]) for v in valid):
        f.unlink(); dropped += 1
print(f"{len(rows)} models, dropped {dropped} orphan prediction files")
