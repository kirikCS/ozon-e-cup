"""Собрать один zip для передачи сокоманднику: код, измеренные векторы, реестр,
BTYD-кэш, журналы, документы. Без cache/ (пересобирается) и data/ (у него есть)."""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ecup_handover_2026-08-14.zip"

INCLUDE = [
    ("code", "*.py"),
    ("docs", "*"),
    ("results", "*"),
    ("submissions", "*.csv"),
    ("submission/components", "*.csv"),
    ("submission/queue", "*.csv"),
]
SINGLES = [
    "text.md", "AGENT.md", "CLAUDE.md",
    "model_out/stack_constraints.json",
    "model_out/btyd_feats.pkl",
    "model_out/btyd_params.json",
]

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for d, pat in INCLUDE:
        for f in sorted((ROOT / d).glob(pat)):
            if f.is_file() and "__pycache__" not in str(f):
                z.write(f, f.relative_to(ROOT))
                n += 1
    for s in SINGLES:
        f = ROOT / s
        if f.exists():
            z.write(f, s)
            n += 1
    for f in sorted((ROOT / "model_out").glob("t*_probe.json")):
        z.write(f, f.relative_to(ROOT))
        n += 1

print(f"{OUT.name}: {n} files, {OUT.stat().st_size/1e6:.1f} MB")
