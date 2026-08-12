"""Drop scoreboard rows by substring, keeping the last occurrence of each model name.

Usage: python models/clean_sb.py model_out/scoreboard.jsonl [substring ...]
Rows whose `model` contains any given substring are removed. With no substrings it only dedupes.
"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
subs = sys.argv[2:]
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows = [r for r in rows if not any(s in r["model"] for s in subs)]
seen = {}
for r in rows:
    seen[r["model"]] = r                     # last write wins
out = list(seen.values())
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print(f"kept {len(out)}")
