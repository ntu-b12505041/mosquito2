from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


METRIC_KEYS = ["family", "model", "lead_mode", "preprocess", "sampling_rate", "max_records"]


def write_metrics(rows: list[dict[str, object]], csv_path: str | Path, json_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    new_df = pd.DataFrame(rows)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    else:
        df = new_df
    keys = [key for key in METRIC_KEYS if key in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return df

