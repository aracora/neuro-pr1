from __future__ import annotations

import numpy as np
import pandas as pd

from labels import NODE1_CLASSES


def aggregate_windows(df: pd.DataFrame, numeric_cols, categorical_cols) -> pd.DataFrame:
    records: list[dict] = []
    for win_id, g in df.groupby("window_id", sort=False):
        g = g.reset_index(drop=True)
        y = g["label"].to_numpy(dtype=np.int64)
        counts = np.bincount(y, minlength=len(NODE1_CLASSES))
        label = int(counts.argmax())
        row: dict = {"window_id": win_id, "label": label}

        num = g[numeric_cols].to_numpy(dtype=np.float64)
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        row_mean = num.mean(axis=0); row_std = num.std(axis=0)
        row_min = num.min(axis=0); row_max = num.max(axis=0)
        for i, col in enumerate(numeric_cols):
            row[f"{col}__mean"] = float(row_mean[i])
            row[f"{col}__std"] = float(row_std[i])
            row[f"{col}__min"] = float(row_min[i])
            row[f"{col}__max"] = float(row_max[i])

        for col in categorical_cols:
            vals = pd.to_numeric(g[col], errors="coerce").fillna(-1).astype(np.int64)
            row[f"{col}__mode"] = int(vals.mode().iloc[0]) if len(vals) else -1
        records.append(row)
    return pd.DataFrame(records)


def feature_columns(numeric_cols, categorical_cols) -> list[str]:
    cols: list[str] = []
    for c in numeric_cols:
        cols += [f"{c}__mean", f"{c}__std", f"{c}__min", f"{c}__max"]
    for c in categorical_cols:
        cols.append(f"{c}__mode")
    return cols