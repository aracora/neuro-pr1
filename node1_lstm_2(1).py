#!/usr/bin/env python
"""Узел 2 — BiLSTM. Потоковое чтение Parquet без загрузки 13 млн строк в RAM.

Предсказание делается для окна; нормализованная матрица ошибок считается
по КАЖДОМУ flow с исходной меткой (предсказание окна распространяется на flow).
Функции LSTMClassifier/make_sequences/SequenceCollator сохранены для pipeline.py.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from labels_node2 import NODE1_CLASSES, CLASS_NAMES_RU

SEED = 42
PROCESSED_DIR = Path("data/processed_node2")
MODEL_DIR = Path("models/node1_lstm_2")
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.25
BIDIRECTIONAL = True
PORT_EMBED_DIM = 8
PROTOCOL_EMBED_DIM = 4
MAX_FLOWS_PER_WINDOW = 128  # 256-flow окно -> 128 равномерных отсчётов для LSTM
FOCAL_GAMMA = 2.0
FORCED_CATEGORICAL = ["Source Port", "Destination Port", "Protocol"]
READ_BATCH_ROWS = 65536


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    total = len(y)
    return np.array([total / (n_classes * c) if c > 0 else 1.0 for c in counts], dtype=np.float32)


def load_split(name: str) -> pd.DataFrame:
    """Совместимость со старым API; для ОБУЧЕНИЯ не используется!"""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet").reset_index(drop=True)


def _parquet_file(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Не установлен pyarrow: pip install pyarrow") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Не найден {path}. Сначала выполните preprocess_node2.py")
    return pq.ParquetFile(path)


def _row_count(path: Path) -> int:
    return _parquet_file(path).metadata.num_rows


def _iter_batches(path: Path, columns):
    """Читает только необходимые колонки, не целиком весь parquet."""
    pf = _parquet_file(path)
    missing = set(columns) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"В {path} отсутствуют колонки: {sorted(missing)}")
    for batch in pf.iter_batches(columns=list(dict.fromkeys(columns)),
                                 batch_size=READ_BATCH_ROWS, use_threads=True):
        yield batch.to_pandas()


def _numerical_array(df: pd.DataFrame, numeric_cols):
    arr = df[numeric_cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _category_array(series):
    return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def build_vocab(series):
    values = _category_array(series)
    return {int(v): i + 1 for i, v in enumerate(np.unique(values))}


def _learn_scaler_and_vocab(path: Path, numeric_cols):
    """Один последовательный проход train, StandardScaler.partial_fit и словари портов."""
    scaler = StandardScaler()
    seen = {name: set() for name in FORCED_CATEGORICAL}
    columns = numeric_cols + FORCED_CATEGORICAL
    print(f"[1/3] Статистика и словари TRAIN: {path} ({_row_count(path):,} flow)", flush=True)
    with tqdm(total=_row_count(path), desc="Статистика train", unit="flow", unit_scale=True) as bar:
        for df in _iter_batches(path, columns):
            scaler.partial_fit(_numerical_array(df, numeric_cols))
            for col in FORCED_CATEGORICAL:
                seen[col].update(int(v) for v in np.unique(_category_array(df[col])))
            bar.update(len(df))
    vocabs = [{v: i + 1 for i, v in enumerate(sorted(seen[name]))}
              for name in FORCED_CATEGORICAL]
    return scaler, vocabs


def _iter_groups(path: Path, columns):
    """Окна уже отсортированы preprocess_node*.py по window_id.

    Хвост окна переносится между батчами: окно никогда не разрывается на два.
    """
    tail = None
    seen_ids = set()
    for df in _iter_batches(path, columns):
        if tail is not None:
            df = pd.concat([tail, df], ignore_index=True)
            tail = None
        if len(df) == 0:
            continue
        keys = df["window_id"].to_numpy()
        boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        start = 0
        for end in boundaries:
            key = str(keys[start])
            if key in seen_ids:
                raise ValueError("window_id встречается повторно не подряд: " + key +
                                 ". Пересоздайте данные через preprocess_node*.py")
            seen_ids.add(key)
            yield df.iloc[start:end]
            start = end
        tail = df.iloc[start:].copy()
    if tail is not None and len(tail):
        key = str(tail["window_id"].iloc[0])
        if key in seen_ids:
            raise ValueError("Повторный window_id в конце файла: " + key)
        yield tail


def _group_to_sequence(g, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab,
                       max_flows=MAX_FLOWS_PER_WINDOW, compact=False):
    n = len(g)
    idx = SequenceCollator._select_indices(n, max_flows)
    selected = g.iloc[idx]
    num = scaler.transform(_numerical_array(selected, numeric_cols)).astype(np.float32)
    # Для 13 млн flow храним готовые последовательности в 2 раза компактнее.
    # Перед входом в PyTorch значения снова переводятся в float32.
    if compact:
        num = np.clip(num, -65504, 65504).astype(np.float16)
    src = np.array([src_vocab.get(int(v), 0) for v in _category_array(selected["Source Port"])], dtype=np.int32)
    dst = np.array([dst_vocab.get(int(v), 0) for v in _category_array(selected["Destination Port"])], dtype=np.int32)
    proto = np.array([proto_vocab.get(int(v), 0) for v in _category_array(selected["Protocol"])], dtype=np.int32)
    return num, src, dst, proto


def make_sequences(df, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab):
    """Старый интерфейс для pipeline.py — предсказания в порядке групп window_id."""
    seq, labels = [], []
    for _, g in df.groupby("window_id", sort=False):
        y = g["label"].to_numpy(dtype=np.int64)
        labels.append(int(np.bincount(y, minlength=len(NODE1_CLASSES)).argmax()))
        seq.append(_group_to_sequence(g, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab,
                                      max_flows=len(g), compact=False))
    return seq, np.asarray(labels, dtype=np.int64)


def _load_sequence_split(name, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab):
    """Формируем последовательности потоково; нет огромного train_df в RAM."""
    path = PROCESSED_DIR / f"{name}.parquet"
    count = _row_count(path)
    print(f"[2/3] Последовательности {name.upper()}: {count:,} flow", flush=True)
    columns = ["window_id", "label"] + numeric_cols + FORCED_CATEGORICAL
    sequences, window_labels, true_flow_counts = [], [], []
    with tqdm(total=count, desc=f"Окна {name}", unit="flow", unit_scale=True) as bar:
        for g in _iter_groups(path, columns):
            y = g["label"].to_numpy(dtype=np.int64)
            counts = np.bincount(y, minlength=len(NODE1_CLASSES))
            if len(counts) != len(NODE1_CLASSES):
                raise ValueError(f"Некорректная метка в {name}: {y.min()}..{y.max()}")
            window_labels.append(int(counts.argmax()))
            true_flow_counts.append(counts)
            sequences.append(_group_to_sequence(g, numeric_cols, scaler, src_vocab, dst_vocab,
                                                proto_vocab, compact=True))
            bar.update(len(g))
            if len(sequences) % 1000 == 0:
                bar.set_postfix(окон=f"{len(sequences):,}")
    if not sequences:
        raise ValueError(f"{path} не содержит окон. Проверьте результат preprocess")
    print(f"  {name}: {len(sequences):,} окон, {count:,} flow", flush=True)
    return (sequences, np.asarray(window_labels, dtype=np.int64),
            np.asarray(true_flow_counts, dtype=np.int64))


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences, self.labels = sequences, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (*self.sequences[idx], int(self.labels[idx]))


class SequenceCollator:
    def __init__(self, max_len=MAX_FLOWS_PER_WINDOW):
        self.max_len = max_len

    @staticmethod
    def _select_indices(length, max_len):
        if length <= max_len:
            return np.arange(length)
        return np.linspace(0, length - 1, max_len, dtype=np.int64)

    def __call__(self, batch):
        chosen = []
        for num, src, dst, proto, y in batch:
            idx = self._select_indices(len(num), self.max_len)
            chosen.append((num[idx], src[idx], dst[idx], proto[idx], y))
        lengths = [len(x[0]) for x in chosen]
        max_len = max(lengths)
        n_num = chosen[0][0].shape[1]
        num = np.zeros((len(chosen), max_len, n_num), dtype=np.float32)
        src = np.zeros((len(chosen), max_len), dtype=np.int64)
        dst = np.zeros((len(chosen), max_len), dtype=np.int64)
        proto = np.zeros((len(chosen), max_len), dtype=np.int64)
        y = np.zeros(len(chosen), dtype=np.int64)
        for i, (a, b, c, d, t) in enumerate(chosen):
            L = len(a)
            num[i, :L] = a
            src[i, :L] = b
            dst[i, :L] = c
            proto[i, :L] = d
            y[i] = t
        return (torch.from_numpy(num), torch.from_numpy(src), torch.from_numpy(dst),
                torch.from_numpy(proto), torch.tensor(lengths, dtype=torch.long), torch.from_numpy(y))


class LSTMClassifier(nn.Module):
    def __init__(self, n_numeric, src_cardinality, dst_cardinality, protocol_cardinality,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
                 bidirectional=BIDIRECTIONAL):
        super().__init__()
        self.src_embedding = nn.Embedding(src_cardinality, PORT_EMBED_DIM)
        self.dst_embedding = nn.Embedding(dst_cardinality, PORT_EMBED_DIM)
        self.protocol_embedding = nn.Embedding(protocol_cardinality, PROTOCOL_EMBED_DIM)
        input_size = n_numeric + PORT_EMBED_DIM * 2 + PROTOCOL_EMBED_DIM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_size)
        self.head = nn.Sequential(nn.Linear(out_size, hidden_size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_size, len(NODE1_CLASSES)))

    def forward(self, num, src, dst, protocol, lengths):
        x = torch.cat([num, self.src_embedding(src), self.dst_embedding(dst),
                       self.protocol_embedding(protocol)], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.lstm.bidirectional else h_n[-1]
        return self.head(self.norm(h))


class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum().clamp_min(1e-8)
        return loss.mean()


def evaluate_loader(model, loader, device, description=None):
    model.eval()
    preds, targets = [], []
    iterable = tqdm(loader, desc=description, unit="batch", leave=False) if description else loader
    with torch.no_grad():
        for num, src, dst, proto, lengths, y in iterable:
            logits = model(num.to(device), src.to(device), dst.to(device),
                           proto.to(device), lengths)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    if not preds:
        raise ValueError("Пустая выборка: невозможно оценить модель")
    pred, true = np.concatenate(preds), np.concatenate(targets)
    return float(accuracy_score(true, pred)), pred, true


def flow_confusion_from_window_predictions(flow_counts, predictions):
    """Для каждого окна раздаёт его прогноз каждому исходному flow.

    flow_counts[i, k] — сколько flow истинного класса k в окне i.
    Это точный эквивалент confusion_matrix по всем исходным строкам без
    создания гигантских массивов из миллионов повторных предсказаний.
    """
    pred = np.asarray(predictions, dtype=np.int64)
    counts = np.asarray(flow_counts, dtype=np.int64)
    n_cls = len(NODE1_CLASSES)
    if len(pred) != len(counts):
        raise ValueError("Порядок окон test и предсказаний не совпадает")
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for cls in range(n_cls):
        cm[:, cls] = counts[pred == cls].sum(axis=0)
    if cm.sum() != counts.sum():
        raise AssertionError("При подсчёте потоков потеряны строки")
    return cm


def plot_confusion_matrices(cm, classes_ru, out_dir: Path):
    """Только нормализованная по истинным классам матрица (в процентах)."""
    totals = cm.sum(axis=1, keepdims=True)
    cm_pct = 100 * np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals != 0)
    size = max(6, len(classes_ru) * 1.2)
    fig, ax = plt.subplots(figsize=(size + .6, size * .85))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
                xticklabels=classes_ru, yticklabels=classes_ru,
                cbar_kws={"label": "Доля потоков, %"}, ax=ax)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"Нормализованная матрица — узел 2 (LSTM)\n"
                 f"Верно {np.trace(cm):,} из {cm.sum():,} потоков")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "node1_lstm_2_confusion_matrix_normalized.png", dpi=160)
    plt.close(fig)
    # Старые файлы иначе останутся в папке и создадут впечатление,
    # будто модель по-прежнему строит ненормализованные матрицы.
    (out_dir / "node1_lstm_2_confusion_matrix.png").unlink(missing_ok=True)


def plot_training_curves(history, out_dir: Path):
    epochs = [v["epoch"] for v in history]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, [v["train_accuracy"] for v in history], label="Обучающая выборка", linewidth=2)
    ax.plot(epochs, [v["val_accuracy"] for v in history], label="Валидационная выборка", linewidth=2)
    ax.set(xlabel="Эпоха", ylabel="Accuracy", title="Узел 2 — LSTM: точность по окнам")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "node1_lstm_2_accuracy.png", dpi=160)
    plt.close(fig)
    (out_dir / "node1_lstm_2_loss.png").unlink(missing_ok=True)


def flow_metrics(cm, classes_ru):
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    hits = np.diag(cm)
    precision = np.divide(hits, predicted, out=np.zeros(len(hits), dtype=float), where=predicted > 0)
    recall = np.divide(hits, support, out=np.zeros(len(hits), dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(hits), dtype=float), where=(precision + recall) > 0)
    n = int(cm.sum())
    accuracy = float(np.trace(cm) / n) if n else 0.0
    # То же определение macro-F1, что и sklearn при labels=range(n_classes).
    macro_f1 = float(f1.mean()) if len(f1) else 0.0
    weighted_f1 = float(np.average(f1, weights=support)) if n else 0.0
    lines = [f"{'Класс':<17} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Flow':>12}"]
    for name, p, r, f, s in zip(classes_ru, precision, recall, f1, support):
        lines.append(f"{name:<17} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>12,}")
    return {"accuracy": accuracy, "f1_macro": macro_f1, "f1_weighted": weighted_f1,
            "classification_report": "\n".join(lines)}


def main():
    parser = argparse.ArgumentParser(description="BiLSTM с потоковой подготовкой parquet")
    parser.add_argument("--fresh", action="store_true", help="Запустить обучение заново (флаг совместимости)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs и --batch-size должны быть положительными")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}", flush=True)

    feature_path = PROCESSED_DIR / "feature_names.json"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Не найден {feature_path}; запустите предобработку")
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    numeric_cols = [col for col in features if col not in FORCED_CATEGORICAL]
    if not numeric_cols:
        raise ValueError("Нет числовых признаков для LSTM")

    scaler, vocabs = _learn_scaler_and_vocab(PROCESSED_DIR / "train.parquet", numeric_cols)
    src_vocab, dst_vocab, proto_vocab = vocabs
    train_seq, y_train, train_counts = _load_sequence_split(
        "train", numeric_cols, scaler, *vocabs)
    val_seq, y_val, val_counts = _load_sequence_split(
        "val", numeric_cols, scaler, *vocabs)
    test_seq, y_test, test_counts = _load_sequence_split(
        "test", numeric_cols, scaler, *vocabs)
    print("[3/3] Подготовка завершена — начинаю обучение LSTM", flush=True)

    class_counts = np.bincount(y_train, minlength=len(NODE1_CLASSES)).astype(np.float64)
    sample_weights = (1.0 / np.maximum(class_counts, 1.0))[y_train]
    sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double),
                                    num_samples=len(y_train), replacement=True)
    collator = SequenceCollator()
    pin_memory = device.type == "cuda"
    train_set = SequenceDataset(train_seq, y_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=collator, pin_memory=pin_memory)
    # Для train accuracy нельзя использовать взвешенный случайный sampler:
    # метрика должна считаться на реальной обучающей выборке.
    train_eval_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=collator, pin_memory=pin_memory)
    val_loader = DataLoader(SequenceDataset(val_seq, y_val), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator, pin_memory=pin_memory)
    test_loader = DataLoader(SequenceDataset(test_seq, y_test), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collator, pin_memory=pin_memory)

    weights = torch.tensor(balanced_weights(y_train, len(NODE1_CLASSES)),
                           dtype=torch.float32, device=device)
    criterion = FocalLoss(weight=weights)
    model = LSTMClassifier(n_numeric=len(numeric_cols), src_cardinality=len(src_vocab) + 1,
                           dst_cardinality=len(dst_vocab) + 1,
                           protocol_cardinality=len(proto_vocab) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history, best_state, best_f1, wait = [], None, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum, observed = 0.0, 0
        for num, src, dst, proto, lengths, y in tqdm(
                train_loader, desc=f"Эпоха {epoch}/{args.epochs}", unit="batch"):
            num = num.to(device, non_blocking=True)
            src = src.to(device, non_blocking=True)
            dst = dst.to(device, non_blocking=True)
            proto = proto.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(num, src, dst, proto, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            observed += len(y)

        train_acc, _, _ = evaluate_loader(model, train_eval_loader, device, "Проверка train")
        val_acc, val_pred, val_true = evaluate_loader(model, val_loader, device, "Проверка val")
        val_f1 = float(f1_score(val_true, val_pred, labels=list(range(len(NODE1_CLASSES))),
                                average="macro", zero_division=0))
        train_loss = loss_sum / max(observed, 1)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "train_accuracy": round(train_acc, 6), "val_accuracy": round(val_acc, 6),
                        "val_macro_f1": round(val_f1, 6)})
        print(f"Эпоха {epoch:>2}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
              f"val_F1={val_f1:.4f}, loss={train_loss:.4f}, "
              f"время={time.perf_counter() - epoch_start:.1f} с", flush=True)
        if val_f1 > best_f1 + 1e-5:
            best_f1, best_state, wait = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping после {epoch} эпох", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(), "classes": NODE1_CLASSES, "classes_ru": CLASS_NAMES_RU,
        "features": features, "numeric_cols": numeric_cols,
        "categorical_cols": FORCED_CATEGORICAL,
        "src_vocab": src_vocab, "dst_vocab": dst_vocab, "protocol_vocab": proto_vocab,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "config": {"model": "BiLSTM", "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
                   "dropout": DROPOUT, "max_flows_per_window": MAX_FLOWS_PER_WINDOW,
                   "focal_gamma": FOCAL_GAMMA},
    }, MODEL_DIR / "model.pt")

    test_acc, test_pred, test_true = evaluate_loader(model, test_loader, device, "Проверка test")
    cm_flow = flow_confusion_from_window_predictions(test_counts, test_pred)
    metrics = flow_metrics(cm_flow, CLASS_NAMES_RU)
    result = {"node": 2, "model": "BiLSTM", "classes": NODE1_CLASSES,
              "classes_ru": CLASS_NAMES_RU, "macro_f1": metrics["f1_macro"],
              "accuracy": metrics["accuracy"], "f1_weighted": metrics["f1_weighted"],
              "accuracy_windows": test_acc, "best_val_macro_f1": best_f1,
              "test_flows": int(cm_flow.sum()), "test_windows": len(test_true)}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (MODEL_DIR / "training_history.json").write_text(
        json.dumps({"history": history}, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_confusion_matrices(cm_flow, CLASS_NAMES_RU, MODEL_DIR)
    plot_training_curves(history, MODEL_DIR)
    print(f"\nTest по окнам: Accuracy={test_acc:.4f}")
    print(f"Test по flow: Accuracy={metrics['accuracy']:.4f}, "
          f"F1 (macro)={metrics['f1_macro']:.4f} (всего {cm_flow.sum():,} flow)")
    print(metrics["classification_report"], flush=True)


if __name__ == "__main__":
    main()
