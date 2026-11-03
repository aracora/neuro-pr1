#!/usr/bin/env python
"""Узел 1 — ансамбль LSTM + MLP + LightGBM.

Классы: SYN, UDP, LDAP, DNS, BENIGN (нормальный трафик).
Hard voting, soft voting, weighted soft voting (веса из val-F1).

Запуск:
    python preprocess.py --raw data/raw --out data/processed --window 5
    python node1_lstm.py --fresh
    python node1_mlp.py --fresh
    python node1_lgbm.py
    python node1_ensemble.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from labels import NODE1_CLASSES, CLASS_LABELS_RU, CLASS_NAMES_RU
from window_features import aggregate_windows, feature_columns

SEED = 42
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
ENSEMBLE_DIR = MODEL_DIR / "node1_ensemble"

PORT_EMBED_DIM = 8
PROTOCOL_EMBED_DIM = 4
MAX_FLOWS_PER_WINDOW = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.25
BIDIRECTIONAL = True
FORCED_CATEGORICAL = ["Source Port", "Destination Port", "Protocol"]


def evaluate(y_true, y_pred, classes_ru):
    n_classes = len(classes_ru); labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                      target_names=classes_ru, zero_division=0),
        "confusion_matrix": cm,
    }


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


class LSTMDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return (*self.sequences[idx], int(self.labels[idx]))


class LSTMSequenceCollator:
    def __init__(self, max_len=MAX_FLOWS_PER_WINDOW): self.max_len = max_len
    @staticmethod
    def _select_indices(length, max_len):
        if length <= max_len: return np.arange(length)
        return np.linspace(0, length - 1, max_len, dtype=np.int64)
    def __call__(self, batch):
        chosen = []
        for num, src, dst, proto, y in batch:
            idx = self._select_indices(len(num), self.max_len)
            chosen.append((num[idx], src[idx], dst[idx], proto[idx], y))
        lengths = [len(x[0]) for x in chosen]
        max_len = max(lengths); n_num = chosen[0][0].shape[1]
        num = np.zeros((len(chosen), max_len, n_num), dtype=np.float32)
        src = np.zeros((len(chosen), max_len), dtype=np.int64)
        dst = np.zeros((len(chosen), max_len), dtype=np.int64)
        proto = np.zeros((len(chosen), max_len), dtype=np.int64)
        y = np.zeros(len(chosen), dtype=np.int64)
        for i, (a, b, c, d, t) in enumerate(chosen):
            L = len(a)
            num[i, :L] = a; src[i, :L] = b; dst[i, :L] = c; proto[i, :L] = d; y[i] = t
        return (torch.from_numpy(num), torch.from_numpy(src), torch.from_numpy(dst),
                torch.from_numpy(proto), torch.tensor(lengths, dtype=torch.long),
                torch.from_numpy(y))


def predict_lstm(model_dir: Path, test_df):
    device = torch.device("cpu")
    ckpt = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    numeric_cols = ckpt["numeric_cols"]
    src_vocab = {int(k): int(v) for k, v in ckpt["src_vocab"].items()}
    dst_vocab = {int(k): int(v) for k, v in ckpt["dst_vocab"].items()}
    proto_vocab = {int(k): int(v) for k, v in ckpt["protocol_vocab"].items()}

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2; scaler.n_features_in_ = len(numeric_cols)

    sequences, labels, window_ids = [], [], []
    df = test_df.sort_values("window_id", kind="stable")
    for win_id, g in df.groupby("window_id", sort=True):
        g = g.reset_index(drop=True)
        y = g["label"].to_numpy(dtype=np.int64)
        counts = np.bincount(y, minlength=len(NODE1_CLASSES))
        label = int(counts.argmax())
        num = g[numeric_cols].to_numpy(dtype=np.float32)
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        num = scaler.transform(num).astype(np.float32)
        src_raw = pd.to_numeric(g["Source Port"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        dst_raw = pd.to_numeric(g["Destination Port"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        proto_raw = pd.to_numeric(g["Protocol"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        src = np.array([src_vocab.get(int(v), 0) for v in src_raw], dtype=np.int64)
        dst = np.array([dst_vocab.get(int(v), 0) for v in dst_raw], dtype=np.int64)
        proto = np.array([proto_vocab.get(int(v), 0) for v in proto_raw], dtype=np.int64)
        sequences.append((num, src, dst, proto)); labels.append(label); window_ids.append(win_id)

    model = LSTMClassifier(n_numeric=len(numeric_cols),
                           src_cardinality=len(src_vocab) + 1,
                           dst_cardinality=len(dst_vocab) + 1,
                           protocol_cardinality=len(proto_vocab) + 1).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    loader = DataLoader(LSTMDataset(sequences, labels), batch_size=64, shuffle=False,
                        num_workers=0, collate_fn=LSTMSequenceCollator())
    probs_all = []
    with torch.no_grad():
        for num, src, dst, proto, lengths, y in loader:
            logits = model(num.to(device), src.to(device), dst.to(device),
                           proto.to(device), lengths.to(device))
            probs_all.append(F.softmax(logits, dim=-1).cpu().numpy())
    return window_ids, np.asarray(labels, dtype=np.int64), np.concatenate(probs_all, axis=0)


class MLPClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_sizes, dropout):
        super().__init__()
        layers = []; prev = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def predict_mlp(model_dir: Path, test_df):
    device = torch.device("cpu")
    ckpt = torch.load(model_dir / "model.pt", map_location=device, weights_only=False)
    feat_cols = ckpt["feature_columns"]; numeric_cols = ckpt["numeric_cols"]
    categorical_cols = ckpt["categorical_cols"]
    hidden_sizes = tuple(ckpt["config"]["hidden_sizes"]); dropout = float(ckpt["config"]["dropout"])

    agg = aggregate_windows(test_df, numeric_cols, categorical_cols)
    agg = agg.sort_values("window_id", kind="stable").reset_index(drop=True)
    X = agg[feat_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2; scaler.n_features_in_ = X.shape[1]
    X = scaler.transform(X).astype(np.float32)

    model = MLPClassifier(X.shape[1], len(NODE1_CLASSES), hidden_sizes, dropout).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    with torch.no_grad():
        probs = F.softmax(model(torch.from_numpy(X).to(device)), dim=-1).cpu().numpy()
    return agg["window_id"].tolist(), agg["label"].to_numpy(dtype=np.int64), probs


def predict_lgbm(model_dir: Path, test_df):
    import lightgbm as lgb
    with open(model_dir / "feature_columns.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(PROCESSED_DIR / "feature_names.json", encoding="utf-8") as f:
        features = json.load(f)
    numeric_cols = [c for c in features if c not in FORCED_CATEGORICAL]
    categorical_cols = list(FORCED_CATEGORICAL)

    agg = aggregate_windows(test_df, numeric_cols, categorical_cols)
    agg = agg.sort_values("window_id", kind="stable").reset_index(drop=True)
    X = agg[feat_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    probs = booster.predict(X)
    if probs.ndim == 1:
        probs = np.stack([1 - probs, probs], axis=1)
    return agg["window_id"].tolist(), agg["label"].to_numpy(dtype=np.int64), probs


def align_predictions(model_preds):
    common = None
    for _, (wids, _, _) in model_preds.items():
        s = set(wids); common = s if common is None else (common & s)
    common_sorted = sorted(common)
    y_true_ref = None; probs = {}
    for name, (wids, y, p) in model_preds.items():
        idx = {w: i for i, w in enumerate(wids)}
        order = [idx[w] for w in common_sorted]
        probs[name] = p[order]
        if y_true_ref is None: y_true_ref = y[order]
    return common_sorted, y_true_ref, probs


def hard_voting(probs):
    votes = np.stack([p.argmax(axis=1) for p in probs.values()], axis=1)
    out = np.zeros(votes.shape[0], dtype=np.int64)
    for i in range(votes.shape[0]):
        out[i] = np.bincount(votes[i], minlength=len(NODE1_CLASSES)).argmax()
    return out


def soft_voting(probs, weights=None):
    names = list(probs.keys())
    P = np.stack([probs[n] for n in names], axis=0)
    if weights is None:
        weights = np.ones(len(names)) / len(names)
    else:
        weights = np.asarray(weights, dtype=float); weights = weights / weights.sum()
    return np.tensordot(weights, P, axes=([0], [0])).argmax(axis=1)


def plot_confusion(cm, classes_ru, out_dir: Path, title, fname, cmap="Blues", fmt="d"):
    size = max(6, len(classes_ru) * 1.2)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=classes_ru, yticklabels=classes_ru, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title(title)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150); plt.close()


def plot_comparison_bar(results, out_dir: Path):
    names = list(results.keys()); vals = [results[n] for n in names]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(names)])
    plt.ylim(0, 1.0); plt.ylabel("macro-F1")
    plt.title("Сравнение моделей и ансамблей")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    plt.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "node1_ensemble_comparison.png", dpi=150); plt.close()


def main():
    print("=== Ансамбль: LSTM + MLP + LightGBM ===\n")
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    print(f"Test: {test_df.window_id.nunique()} окон / {len(test_df):,} flow\n")

    model_dirs = {"LSTM": MODEL_DIR / "node1_lstm", "MLP": MODEL_DIR / "node1_mlp",
                  "LightGBM": MODEL_DIR / "node1_lgbm"}
    for name, d in model_dirs.items():
        if not d.exists():
            raise SystemExit(f"Нет папки {d}. Сначала обучи модель.")

    print("LSTM ...")
    lstm_pred = predict_lstm(model_dirs["LSTM"], test_df)
    print(f"  окон: {len(lstm_pred[0])}, probs.shape={lstm_pred[2].shape}")
    print("MLP ...")
    mlp_pred = predict_mlp(model_dirs["MLP"], test_df)
    print(f"  окон: {len(mlp_pred[0])}, probs.shape={mlp_pred[2].shape}")
    print("LightGBM ...")
    lgbm_pred = predict_lgbm(model_dirs["LightGBM"], test_df)
    print(f"  окон: {len(lgbm_pred[0])}, probs.shape={lgbm_pred[2].shape}")

    preds = {"LSTM": lstm_pred, "MLP": mlp_pred, "LightGBM": lgbm_pred}
    wids, y_true, probs = align_predictions(preds)
    print(f"\nОбщий набор окон после выравнивания: {len(wids)}")
    print(f"Классы в test: {[CLASS_LABELS_RU[NODE1_CLASSES[i]] for i in sorted(np.unique(y_true))]}\n")

    results = {}
    for name, p in probs.items():
        pred = p.argmax(axis=1)
        f1 = float(f1_score(y_true, pred, labels=list(range(len(NODE1_CLASSES))),
                            average="macro", zero_division=0))
        results[name] = f1
        print(f"{name:>10s}: accuracy={accuracy_score(y_true, pred):.4f}, macro-F1={f1:.4f}")

    hard_pred = hard_voting(probs)
    hard_f1 = float(f1_score(y_true, hard_pred, labels=list(range(len(NODE1_CLASSES))),
                             average="macro", zero_division=0))
    results["Hard voting"] = hard_f1
    print(f"{'Hard':>10s}: accuracy={accuracy_score(y_true, hard_pred):.4f}, macro-F1={hard_f1:.4f}")

    soft_pred = soft_voting(probs)
    soft_f1 = float(f1_score(y_true, soft_pred, labels=list(range(len(NODE1_CLASSES))),
                             average="macro", zero_division=0))
    results["Soft voting"] = soft_f1
    print(f"{'Soft':>10s}: accuracy={accuracy_score(y_true, soft_pred):.4f}, macro-F1={soft_f1:.4f}")

    val_f1s = [1.0, 1.0, 1.0]
    print(f"\nВеса (равные, т.к. нет валидации): {dict(zip(probs.keys(), val_f1s))}")

    weighted_pred = soft_voting(probs, weights=np.array(val_f1s, dtype=float))
    weighted_f1 = float(f1_score(y_true, weighted_pred, labels=list(range(len(NODE1_CLASSES))),
                                 average="macro", zero_division=0))
    results["Weighted soft"] = weighted_f1
    print(f"{'Weighted':>10s}: accuracy={accuracy_score(y_true, weighted_pred):.4f}, "
          f"macro-F1={weighted_f1:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nЛучший вариант: {best_name} (macro-F1={results[best_name]:.4f})")

    best_pred = {"Soft voting": soft_pred, "Weighted soft": weighted_pred,
                 "Hard voting": hard_pred}.get(best_name)
    if best_pred is not None:
        cm = confusion_matrix(y_true, best_pred, labels=list(range(len(NODE1_CLASSES))))
        plot_confusion(cm, CLASS_NAMES_RU, ENSEMBLE_DIR,
                       f"Матрица ошибок — {best_name}", "node1_ensemble_confusion_matrix.png")
        denom = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
        plot_confusion(cm_norm, CLASS_NAMES_RU, ENSEMBLE_DIR,
                       f"Нормализованная матрица — {best_name}",
                       "node1_ensemble_confusion_matrix_normalized.png", fmt=".2f")

    plot_comparison_bar(results, ENSEMBLE_DIR)
    best_metrics = evaluate(y_true, best_pred, CLASS_NAMES_RU)

    out = {"node": 1, "ensemble": "LSTM + MLP + LightGBM", "method": best_name,
           "n_test_windows": int(len(wids)), "classes": NODE1_CLASSES, "classes_ru": CLASS_NAMES_RU,
           "results_macro_f1": {k: round(v, 6) for k, v in results.items()},
           "val_f1_weights": dict(zip(probs.keys(), [round(x, 6) for x in val_f1s])),
           "best_macro_f1": round(results[best_name], 6),
           "best_accuracy": best_metrics["accuracy"],
           "classification_report": best_metrics["classification_report"]}
    (ENSEMBLE_DIR / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nAccuracy: {best_metrics['accuracy']:.4f}")
    print(f"F1-score (macro): {best_metrics['f1_macro']:.4f}")
    print(f"F1-score (weighted): {best_metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(best_metrics["classification_report"])
    print(f"Артефакты сохранены в {ENSEMBLE_DIR}/")


if __name__ == "__main__":
    main()