"""Rebuild stack_daily.csv from the individual components and the solved weights."""
import json
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
cfg = json.loads((HERE / "weights.json").read_text(encoding="utf-8"))
EL = cfg["EL"]

acc, uid = None, None
for name, w in cfg["weights"].items():
    d = pl.read_csv(HERE / "components" / f"{name}.csv").sort("user_id")
    u = d["user_id"].to_numpy()
    if uid is None:
        uid = u
    assert np.array_equal(u, uid), f"user_id order differs in {name}"
    q = np.log1p(d["predict"].to_numpy())
    q = q - q.mean() + EL                     # level-match, idempotent
    acc = w * q if acc is None else acc + w * q

acc = acc + (EL - acc.mean())                 # the mixture is level-matched once more
pred = np.clip(np.expm1(acc), 0, None)
pl.DataFrame({"user_id": uid, "predict": pred}).write_csv(HERE / "stack_daily.csv")
print(f"stack_daily.csv: {len(pred)} rows, mean log1p {np.log1p(pred).mean():.4f}, "
      f"predicted RMSLE {cfg['predicted_rmsle']:.6f}")
