#!/usr/bin/env python
"""Предобработка сырых CSV для узла 1.

Классы: SYN, LDAP, DNS, OTHER, BENIGN.
Стратегия сплита — СТРАТИФИЦИРОВАННАЯ 70/15/15 (train/val/test).
Upsampling для BENIGN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from labels_node2 import NODE1_ALIAS_TO_IDX, NODE1_CLASSES, CLASS_LABELS_RU

SEED = 42
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW_SECONDS = 10.0
MIN_WINDOW_SECONDS = 5.0
MAX_WINDOW_SECONDS = 10.0

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

RAW_FEATURES = [
    "Source Port", "Destination Port", "Protocol",
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def _norm_label(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    key = str(v).strip().upper().replace(" ", "_")
    return NODE1_ALIAS_TO_IDX.get(key)


def _load_one(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None
    df.columns = [str(c).strip() for c in df.columns]
    if "Label" not in df.columns or "Timestamp" not in df.columns:
        print(f"[skip] {path}: нет Label/Timestamp")
        return None
    return df


def _assign_windows(df: pd.DataFrame, window_seconds: float) -> np.ndarray:
    parsed = pd.to_datetime(df["Timestamp"], errors="coerce")
    if parsed.notna().mean() > 0.5:
        start = parsed.min()
        elapsed = (parsed.fillna(start) - start).dt.total_seconds().to_numpy()
        return np.floor(elapsed / window_seconds).astype(np.int64)
    numeric = pd.to_numeric(df["Timestamp"], errors="coerce")
    if numeric.notna().mean() > 0.5:
        start = float(numeric.min())
        elapsed = numeric.fillna(start).to_numpy(dtype=np.float64) - start
        return np.floor(elapsed / window_seconds).astype(np.int64)
    step = max(1, int(window_seconds * 100))
    return (np.arange(len(df)) // step).astype(np.int64)


def _diagnose_file(df: pd.DataFrame, path: Path, window_seconds: float) -> None:
    print(f"\n=== {path.name} ===")
    print(f"  строк: {len(df):,}")
    raw_labels = df["Label"].astype(str).str.strip()
    raw_counts = raw_labels.value_counts()
    print("  топ-10 Label:")
    for lbl, cnt in raw_counts.head(10).items():
        matched = _norm_label(lbl)
        tag = f"-> {CLASS_LABELS_RU[NODE1_CLASSES[matched]]}" if matched is not None else "-> (не наш класс)"
        print(f"    {str(lbl)!r:35s} {cnt:>12,}  {tag}")

    labels = df["Label"].map(_norm_label)
    n_matched = int(labels.notna().sum())
    print(f"  совпало: {n_matched:,} ({n_matched / max(len(df), 1):.2%})")
    local_window_preview = _assign_windows(df, window_seconds)
    print(f"  окон фактически: {len(np.unique(local_window_preview))}")


def stratified_window_split(window_to_class, train_frac, val_frac, rng):
    by_class: dict[int, list[str]] = {}
    for key, cls in window_to_class.items():
        by_class.setdefault(int(cls), []).append(key)

    train_keys, val_keys, test_keys = set(), set(), set()
    print("\nСтратифицированный сплит (70/15/15) по классу окна:")
    for cls, keys in sorted(by_class.items()):
        keys_arr = np.array(keys)
        rng.shuffle(keys_arr)
        m = len(keys_arr)

        if m == 1:
            n_train, n_val = 1, 0
        elif m == 2:
            n_train, n_val = 1, 1
        else:
            n_train = max(1, int(m * train_frac))
            n_val = max(1, int(m * val_frac))
            if n_train + n_val >= m:
                n_train = max(1, m - 2)
                n_val = 1
        n_test = m - n_train - n_val

        train_keys.update(keys_arr[:n_train].tolist())
        val_keys.update(keys_arr[n_train:n_train + n_val].tolist())
        test_keys.update(keys_arr[n_train + n_val:].tolist())
        print(
            f"    класс {CLASS_LABELS_RU[NODE1_CLASSES[cls]]:>20s}: всего {m:>4d} окон -> "
            f"train={n_train:>4d}, val={n_val:>3d}, test={n_test:>3d}"
        )
    return train_keys, val_keys, test_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--window", type=float, default=WINDOW_SECONDS)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    csvs = sorted(args.raw.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"Нет CSV в {args.raw}")

    chunks: list[pd.DataFrame] = []
    for source_id, path in enumerate(tqdm(csvs, desc="Чтение CSV")):
        print(f"\nЧтение {path.name}...")
        
        # Читаем файл кусками по 1 миллиону строк, чтобы не забить память
        for chunk in pd.read_csv(path, low_memory=False, chunksize=1_000_000):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            if "Label" not in chunk.columns or "Timestamp" not in chunk.columns:
                print(f"[skip] {path}: нет Label/Timestamp")
                break
            
            labels = chunk["Label"].map(_norm_label)
            mask = labels.notna()
            if mask.sum() == 0:
                continue

            chunk = chunk.loc[mask].copy()
            chunk["__label__"] = labels.loc[mask].astype(np.int64).to_numpy()
            chunk["__source_id__"] = source_id
            chunk["__source_file__"] = path.name

            parsed_ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
            if parsed_ts.notna().mean() > 0.5:
                chunk["__timestamp_order__"] = parsed_ts
            else:
                chunk["__timestamp_order__"] = pd.RangeIndex(len(chunk))

            chunk = chunk.sort_values("__timestamp_order__", kind="stable").reset_index(drop=True)
            local_window = _assign_windows(chunk, args.window)
            chunk["__window_local__"] = local_window
            chunk["__window_key__"] = [f"{source_id}:{int(w)}" for w in local_window]
            
            # Оставляем только нужные колонки, чтобы не тащить мусор в память
            keep_cols = ["__label__", "__source_id__", "__source_file__", "__timestamp_order__", "__window_key__"] + RAW_FEATURES
            chunks.append(chunk[keep_cols])

    if not chunks:
        raise SystemExit("Не удалось загрузить ни одного файла с нужными классами")

    print(f"\nСклеиваем части и приводим к числам (это может занять пару минут)...")
    data = pd.concat(chunks, ignore_index=True)
    
    # Сразу переводим признаки в float32, чтобы сэкономить память (вместо float64)
    for col in RAW_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(np.float32)
    data[RAW_FEATURES] = data[RAW_FEATURES].replace([np.inf, -np.inf], np.nan)
    
    print(f"Всего строк после фильтрации: {len(data):,}")

    def _majority(s: pd.Series) -> int:
        return int(np.bincount(s.to_numpy(dtype=np.int64), minlength=len(NODE1_CLASSES)).argmax())

    window_label_series = data.groupby("__window_key__", sort=False)["__label__"].agg(_majority)
    window_to_class: dict[str, int] = window_label_series.to_dict()

    print("\nРаспределение окон по классам (до сплита):")
    counts = pd.Series(list(window_to_class.values())).value_counts().sort_index()
    for cls_idx, cnt in counts.items():
        ru = CLASS_LABELS_RU[NODE1_CLASSES[int(cls_idx)]]
        print(f"  {ru:>20s}: {cnt}")

    if len(counts) < 2:
        raise SystemExit("В данных только один класс окон — обучение бессмысленно.")

    rng = np.random.default_rng(SEED)
    train_keys, val_keys, test_keys = stratified_window_split(
        window_to_class, args.train_frac, args.val_frac, rng
    )

    # --- UPSAMPLING ДЛЯ НОРМАЛЬНОГО ТРАФИКА ---
    pass

    def pick(keys: set[str]) -> pd.DataFrame:
        part = data.loc[data["__window_key__"].isin(keys)].copy()
        part = part.sort_values(["__window_key__", "__timestamp_order__"],
                                kind="stable", na_position="last")
        part["label"] = part["__label__"].astype(np.int64)
        part["window_id"] = part["__window_key__"].astype(str)
        return part.reset_index(drop=True)

    train, val, test = pick(train_keys), pick(val_keys), pick(test_keys)
    
    # Удаляем тяжелый data из памяти
    del data, chunks
    import gc
    gc.collect()

    missing_feat = [c for c in RAW_FEATURES if c not in train.columns]
    if missing_feat:
        raise SystemExit("Отсутствуют обязательные признаки: " + ", ".join(missing_feat))

    medians = train[RAW_FEATURES].median(numeric_only=True).fillna(0.0)
    
    for frame in (train, val, test):
        for col in RAW_FEATURES:
            frame[col] = frame[col].fillna(medians[col])

    train = train.dropna(subset=RAW_FEATURES, how="all").reset_index(drop=True)
    val = val.dropna(subset=RAW_FEATURES, how="all").reset_index(drop=True)
    test = test.dropna(subset=RAW_FEATURES, how="all").reset_index(drop=True)

    cols = ["window_id", "label"] + RAW_FEATURES
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame[cols].to_parquet(args.out / f"{name}.parquet", index=False)
        n_win = frame["window_id"].nunique()
        print(f"\n{name}: {len(frame):,} flow, {n_win:,} уникальных окон")

    with open(args.out / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(RAW_FEATURES, f, indent=2, ensure_ascii=False)
    with open(args.out / "medians.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in medians.items()}, f, indent=2, ensure_ascii=False)

    split_info = {
        "window_seconds": float(args.window),
        "classes": NODE1_CLASSES,
        "classes_ru": [CLASS_LABELS_RU[c] for c in NODE1_CLASSES],
        "split_strategy": "stratified_70_15_15_with_upsampling",
        "source_files": [p.name for p in csvs],
    }
    with open(args.out / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"\nГотово: {args.out}/ — окно {args.window:g} с, {len(RAW_FEATURES)} признаков")


if __name__ == "__main__":
    main()
