"""Stage 6: inject computed data into the report template."""
import json
from pathlib import Path

import polars as pl

OUT = Path("eda_out")
charts = json.load(open(OUT / "chartdata.json"))
s03 = json.load(open(OUT / "s03_summary.json"))

# rename awkward CSV column for JS
charts["seasonal"] = [
    {**r, "sum_target_prev30_ratio": r.pop("sum_target/sum_prev30")}
    for r in pl.read_csv(OUT / "t29_seasonal_calibration.csv").to_dicts()
]

RU = {
    "visit_days (rows per user)": "дней с записью (строк на клиента)",
    "engaged_days": "дней с активностью (флаг канала)",
    "buy_days": "дней с покупкой",
    "lifetime_gmv": "GMV за весь период",
    "lifetime_orders": "заказов за весь период",
    "lifetime_searches": "поисков за весь период",
    "recency_days (from 2026-02-13)": "recency, дней от 2026-02-13",
    "tenure_days (first->last)": "срок жизни, дней (первая → последняя запись)",
}
udist = pl.read_csv(OUT / "t07_user_distributions.csv").to_dicts()
for r in udist:
    r["metric"] = RU.get(r["metric"], r["metric"])

lag = [
    {"lag": r["lag_blocks_30d"], "block": r["block"], "r": r["pearson_log1p_with_last_block"]}
    for r in pl.read_csv(OUT / "t23b_lag_profile.csv").to_dicts()
]
lag = [r for r in lag if r["lag"] >= 1]

data = {
    "charts": charts,
    "summary": {"target": s03["target_at_anchor"], "overview": s03["overview"]},
    "tables": {
        "cols": pl.read_csv(OUT / "t02_column_profile.csv").to_dicts(),
        "udist": udist,
        "lag": lag,
        "yoy": pl.read_csv(OUT / "t25_yoy_seasonality.csv").to_dicts(),
    },
}

payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
assert "</script" not in payload
tpl = Path("eda/report_template.html").read_text()
html = tpl.replace("__DATA__", payload)
Path(OUT / "report.html").write_text(html)
print("report.html:", (OUT / "report.html").stat().st_size / 1024, "KB")
