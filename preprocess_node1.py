#!/usr/bin/env python
"""Предобработка сырых CSV для узла 1.

Классы: BENIGN / ATTACK; дополнительно сохраняется тип атаки для второго узла.
Стратегия сплита — СТРАТИФИЦИРОВАННАЯ 70/15/15 (train/val/test).
Балансировка классов выполняется при обучении моделей.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from labels_node1 import NODE1_ALIAS_TO_IDX, NODE1_CLASSES, CLASS_LABELS_RU
from labels_node2 import NODE1_ALIAS_TO_IDX as ATTACK_ALIAS_TO_IDX

SEED = 42
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

WINDOW_SECONDS = 2.0  # максимальная длительность базового временного интервала
MAX_FLOWS_PER_WINDOW = 256  # ограничение числа flow в одном окне при плотном трафике
MIN_WINDOW_SECONDS = 0.1
MAX_WINDOW_SECONDS = 60.0

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


def _timestamp_seconds(df: pd.DataFrame, timestamp_unit: str = "auto") -> np.ndarray:
    """Приводит Timestamp к секундам, проверяя единицы числового времени.

    Параметр timestamp_unit: auto, s, ms, us, ns, excel-d.
    Временные окна нельзя считать корректными, пока единицы Timestamp неизвестны.
    """
    ts = df["Timestamp"]
    numeric = pd.to_numeric(ts, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        values = numeric.to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError("Timestamp: нет корректных числовых значений")
        magnitude = float(np.median(np.abs(finite)))
        unit = timestamp_unit
        if unit == "auto":
            if 1e7 <= magnitude <= 3e7 and np.all(np.abs(finite[:1000] - np.round(finite[:1000])) < 1e-8):
                # Восемь цифр, например 20181103 — YYYYMMDD, а не секунды!
                sample = str(int(round(magnitude)))
                if len(sample) == 8 and 1900 <= int(sample[:4]) <= 2100:
                    unit = "yyyymmdd"
                else:
                    unit = "s"
            elif magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            elif magnitude >= 1e8:
                unit = "s"
            elif 20000 <= magnitude <= 80000:
                # В большинстве таблиц CICFlowMeter такие даты — числа дней Excel.
                # Если на самом деле это относительные секунды, используйте --timestamp-unit s.
                unit = "excel-d"
            else:
                unit = "s"
        scale = {"s": 1., "ms": 1e3, "us": 1e6, "ns": 1e9}
        if unit in scale:
            seconds = values / scale[unit]
        elif unit == "excel-d":
            seconds = (values - 25569.0) * 86400.0  # Unix-эпоха в системе дат Excel
        elif unit == "yyyymmdd":
            normalized = pd.Series(np.rint(values), index=ts.index).astype("Int64").astype(str)
            parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
            seconds = parsed.astype("int64").to_numpy(dtype=np.float64) / 1e9
            seconds[parsed.isna().to_numpy()] = np.nan
        else:
            raise ValueError(f"Неизвестная единица Timestamp: {unit}")
        if not getattr(_timestamp_seconds, "_unit_reported", False):
            print(f"[Timestamp] числовой формат; интерпретация: {unit}; "
                  f"пример значений: {ts.head(4).tolist()}")
            _timestamp_seconds._unit_reported = True
        return seconds

    if timestamp_unit != "auto":
        raise ValueError("--timestamp-unit задаётся только для числового Timestamp")

    # Для строковых дат сначала пробуем распространённые форматы, затем смешанные.
    variants = []
    for day_first in (False, True):
        parsed = pd.to_datetime(ts, errors="coerce", dayfirst=day_first)
        variants.append(parsed)
        if parsed.notna().mean() < 0.95:
            try:
                variants.append(pd.to_datetime(ts, errors="coerce", dayfirst=day_first, format="mixed"))
            except (TypeError, ValueError):
                pass
    parsed = max(variants, key=lambda p: p.notna().sum())
    if parsed.notna().mean() < 0.95:
        raise ValueError("Timestamp: менее 95% значений удалось разобрать; "
                         f"образцы: {ts.head(5).tolist()}")
    seconds = parsed.astype("int64").to_numpy(dtype=np.float64) / 1e9
    seconds[parsed.isna().to_numpy()] = np.nan
    if not getattr(_timestamp_seconds, "_unit_reported", False):
        print(f"[Timestamp] строковый формат дат; примеры: {ts.head(4).tolist()}")
        _timestamp_seconds._unit_reported = True
    return seconds


def _assign_windows(df: pd.DataFrame, window_seconds: float, timestamp_unit: str = "auto") -> np.ndarray:
    if window_seconds <= 0:
        raise ValueError("Длительность окна должна быть положительной")
    seconds = _timestamp_seconds(df, timestamp_unit=timestamp_unit)
    if not np.isfinite(seconds).all():
        raise ValueError("Timestamp: присутствуют пропуски; исправьте/удалите их до разметки")
    return np.floor(seconds / window_seconds).astype(np.int64)



def _bounded_time_window_keys(
    source_id: int,
    time_bins: np.ndarray,
    max_flows: int,
    bin_cursors: dict[tuple[int, int], int],
) -> list[str]:
    """Делит каждую временную корзину на подокна не более max_flows.

    Счётчики общие для всех чанков одного исходного CSV, поэтому граница
    чанка не создаёт новое подокно и не соединяет разные временные корзины.
    Записи внутри чанка должны быть отсортированы по Timestamp.
    """
    if max_flows <= 0:
        return [f"{source_id}:{int(w)}" for w in time_bins]

    # pandas cumcount линейно рассчитывает позицию flow внутри временной корзины.
    # np.unique позволяет один раз прочитать/обновить счётчики за каждую корзину.
    bins, inverse, counts = np.unique(
        np.asarray(time_bins, dtype=np.int64),
        return_inverse=True,
        return_counts=True,
    )
    bases = np.array(
        [bin_cursors.get((source_id, int(b)), 0) for b in bins],
        dtype=np.int64,
    )
    local_index = pd.Series(time_bins).groupby(time_bins, sort=False).cumcount().to_numpy(dtype=np.int64)
    parts = (bases[inverse] + local_index) // max_flows
    for b, count, base in zip(bins, counts, bases):
        bin_cursors[(source_id, int(b))] = int(base + count)
    return [f"{source_id}:{int(b)}:{int(part)}" for b, part in zip(time_bins, parts)]


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
    ap.add_argument("--window", type=float, default=WINDOW_SECONDS,
                    help="Длительность базового интервала, секунды (по умолчанию: 2)")
    ap.add_argument("--max-flows-per-window", type=int, default=MAX_FLOWS_PER_WINDOW,
                    help="Максимум flow на временное окно (по умолчанию 256; 0 отключает предел)")
    ap.add_argument("--timestamp-unit", choices=["auto", "s", "ms", "us", "ns", "excel-d"], default="auto",
                    help="Единицы числового Timestamp; auto пытается определить по масштабу")
    ap.add_argument("--window-flows", type=int, default=None,
                    help="АЛЬТЕРНАТИВА временным окнам: группировать по N последовательных flow в исходном CSV")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    args = ap.parse_args()
    if args.window_flows is not None and args.window_flows < 2:
        ap.error("--window-flows должен быть не меньше 2")
    if args.max_flows_per_window < 0 or args.max_flows_per_window == 1:
        ap.error("--max-flows-per-window должен быть 0 (отключён) или >= 2")
    if args.window_flows is not None:
        print(f"[РЕЖИМ] окна по {args.window_flows} последовательных flow, а НЕ по времени. "
              "Используйте только если Timestamp слишком грубый / одинаковый.")
    if not (MIN_WINDOW_SECONDS <= args.window <= MAX_WINDOW_SECONDS):
        ap.error(f"--window должен быть в пределах {MIN_WINDOW_SECONDS}..{MAX_WINDOW_SECONDS} секунд")
    if not (0 < args.train_frac < 1 and 0 < args.val_frac < 1 and args.train_frac + args.val_frac < 1):
        ap.error("Доли train/val должны быть положительными, а сумма — меньше 1")

    args.out.mkdir(parents=True, exist_ok=True)
    csvs = sorted(args.raw.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"Нет CSV в {args.raw}")

    chunks: list[pd.DataFrame] = []
    flow_cursor = {}  # счётчик строк после фильтрации по каждому файлу, не сбрасывается на границе чанка
    bin_cursors: dict[tuple[int, int], int] = {}  # предел числа flow сохраняется между чанками
    time_preview = {}
    for source_id, path in enumerate(tqdm(csvs, desc="Чтение CSV")):
        print(f"\nЧтение {path.name}...")
        _timestamp_seconds._unit_reported = False
        
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
            # -1 означает BENIGN; 0..4 — исходный тип атаки (узел 2).
            chunk["__attack_type__"] = (chunk["Label"].astype(str).str.strip()
                .str.upper().str.replace(" ", "_", regex=False)
                .map(ATTACK_ALIAS_TO_IDX).fillna(-1).astype(np.int64))
            if ((chunk["__label__"] == 1) & (chunk["__attack_type__"] < 0)).any():
                raise ValueError("Есть атакующие классы, не представленные в labels_node2.py")
            chunk["__source_id__"] = source_id
            chunk["__source_file__"] = path.name

            # Абсолютные временные окна или явный альтернативный режим по числу flow.
            if args.window_flows is None:
                seconds = _timestamp_seconds(chunk, args.timestamp_unit)
                valid_ts = np.isfinite(seconds)
                if not valid_ts.all():
                    print(f"  пропущено строк с пустым Timestamp: {(~valid_ts).sum():,}")
                    chunk = chunk.loc[valid_ts].copy()
                    seconds = seconds[valid_ts]
                if chunk.empty:
                    continue
                chunk["__timestamp_order__"] = seconds
                chunk = chunk.sort_values("__timestamp_order__", kind="stable").reset_index(drop=True)
                local_window = np.floor(chunk["__timestamp_order__"].to_numpy(dtype=np.float64) / args.window).astype(np.int64)
                window_keys = _bounded_time_window_keys(
                    source_id, local_window, args.max_flows_per_window, bin_cursors
                )
                first, last = float(np.min(seconds)), float(np.max(seconds))
                prev = time_preview.get(source_id)
                time_preview[source_id] = (min(first, prev[0]), max(last, prev[1])) if prev else (first, last)
            else:
                cursor = flow_cursor.get(source_id, 0)
                local_window = np.arange(cursor, cursor + len(chunk), dtype=np.int64) // args.window_flows
                window_keys = [f"{source_id}:{int(w)}" for w in local_window]
                flow_cursor[source_id] = cursor + len(chunk)
                chunk["__timestamp_order__"] = np.arange(cursor, cursor + len(chunk), dtype=np.float64)
                chunk = chunk.reset_index(drop=True)
            chunk["__window_key__"] = window_keys

            # Оставляем только нужные колонки, чтобы не тащить мусор в память
            keep_cols = ["__label__", "__attack_type__", "__source_id__", "__source_file__", "__timestamp_order__", "__window_key__"] + RAW_FEATURES
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

    if args.window_flows is None:
        print("\nОхват времени по исходным CSV:")
        for source_id, (start, end) in sorted(time_preview.items()):
            print(f"  {csvs[source_id].name}: {end - start:,.3f} с, "
                  f"временные границы {start:,.3f} .. {end:,.3f}")
    total_windows = len(window_to_class)
    avg_flows = len(data) / max(total_windows, 1)
    print(f"Итого: {total_windows:,} окон, в среднем {avg_flows:,.1f} flow/окно")
    if args.window_flows is None and args.max_flows_per_window > 0:
        print(f"Временные окна: не более {args.window:g} с и "
              f"не более {args.max_flows_per_window} flow в одном окне")
    if args.window_flows is None and avg_flows > 50_000:
        raise SystemExit(
            "\n[ОШИБКА] По временным меткам получились слишком крупные окна "
            f"({avg_flows:,.0f} flow/окно). Не обучайте модели на этих данных! "
            "Проверьте реальный формат Timestamp; для числовых дат попробуйте "
            "--timestamp-unit s/ms/us/ns/excel-d. Если Timestamp неизменный или слишком "
            "грубый, используйте --window-flows 256 (это окна по числу потоков, НЕ по времени)."
        )

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

    # Балансировка делается внутри моделей: class_weight / WeightedRandomSampler.
    # Прежний UPSAMPLE_FACTOR не работал, так как set.update() не размножает окна.

    def pick(keys: set[str]) -> pd.DataFrame:
        part = data.loc[data["__window_key__"].isin(keys)].copy()
        part = part.sort_values(["__window_key__", "__timestamp_order__"],
                                kind="stable", na_position="last")
        part["label"] = part["__label__"].astype(np.int64)
        part["attack_type"] = part["__attack_type__"].astype(np.int64)
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

    cols = ["window_id", "label", "attack_type"] + RAW_FEATURES
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame[cols].to_parquet(args.out / f"{name}.parquet", index=False)
        n_win = frame["window_id"].nunique()
        print(f"\n{name}: {len(frame):,} flow, {n_win:,} уникальных окон")

    with open(args.out / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(RAW_FEATURES, f, indent=2, ensure_ascii=False)
    with open(args.out / "medians.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in medians.items()}, f, indent=2, ensure_ascii=False)

    split_info = {
        "window_seconds": float(args.window) if args.window_flows is None else None,
        "window_flows": args.window_flows,
        "max_flows_per_window": args.max_flows_per_window if args.window_flows is None else None,
        "window_mode": ("time_with_flow_cap" if args.max_flows_per_window > 0 else "time")
                       if args.window_flows is None else "flow_count",
        "timestamp_unit": args.timestamp_unit,
        "classes": NODE1_CLASSES,
        "classes_ru": [CLASS_LABELS_RU[c] for c in NODE1_CLASSES],
        "split_strategy": "stratified_by_window_class_70_15_15",
        "source_files": [p.name for p in csvs],
    }
    with open(args.out / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    window_description = (f"до {args.window:g} с, максимум {args.max_flows_per_window} flow"
                          if args.window_flows is None and args.max_flows_per_window > 0
                          else f"{args.window:g} с" if args.window_flows is None
                          else f"{args.window_flows} flow")
    print(f"\nГотово: {args.out}/ — окно {window_description}, {len(RAW_FEATURES)} признаков")


if __name__ == "__main__":
    main()