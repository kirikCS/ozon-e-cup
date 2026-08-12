"""Render the experiment scoreboard into a self-contained HTML report."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "model_out")
rows = json.loads((OUT / "summary.json").read_text())
payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
tpl = Path("models/report_template.html").read_text()
dst = OUT / "report_models.html"
dst.write_text(tpl.replace("__DATA__", payload))
print("wrote", dst, dst.stat().st_size // 1024, "KB,", len(rows), "models")
