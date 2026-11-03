#!/usr/bin/env python
"""Узел 1 — LSTM (Бинарная классификация: BENIGN vs ATTACK)."""
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from labels_node1 import NODE1_CLASSES, CLASS_LABELS_RU, CLASS_NAMES_RU

SEED = 42
PROCESSED_DIR = Path("data/processed_node1")
MODEL_DIR = Path("models/node1_lstm_1")

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
MAX_FLOWS_PER_WINDOW = 128
FOCAL_GAMMA = 2.0
FORCED_CATEGORICAL = ["Source Port", "Destination Port", "Protocol"]

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def balanced_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    total = len(y)
    return np.array([total / (n_classes * c) if c > 0 else 1.0 for c in counts], dtype=np.float32)

def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    return pd.read_parquet(path).reset_index(drop=True)

def evaluate(y_true, y_pred, classes_ru):
    n_classes = len(classes_ru); labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=classes_ru, zero_division=0),
        "confusion_matrix": cm,
    }

def plot_confusion_matrices(cm, classes_ru, out_dir: Path):
    size = max(6, len(classes_ru) * 1.2)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes_ru, yticklabels=classes_ru, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок — Узел 1 (LSTM)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / "node1_lstm_1_confusion_matrix.png", dpi=150); plt.close()

    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes_ru, yticklabels=classes_ru, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title("Нормализованная матрица ошибок — Узел 1 (LSTM)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / "node1_lstm_1_confusion_matrix_normalized.png", dpi=150); plt.close()

def plot_training_curves(history, out_dir: Path):
    epochs = [e["epoch"] for e in history]
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [e["train_loss"] for e in history], label="Focal loss (train)")
    plt.xlabel("Эпоха"); plt.ylabel("Loss")
    plt.title("Процесс обучения Узла 1 — LSTM (потери)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "node1_lstm_1_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [e["train_accuracy"] for e in history], label="Точность (train)")
    plt.xlabel("Эпоха"); plt.ylabel("Точность")
    plt.title("Процесс обучения Узла 1 — LSTM (точность)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "node1_lstm_1_accuracy.png", dpi=150); plt.close()

def build_vocab(series):
    values = pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int64).unique().tolist()
    return {v: i + 1 for i, v in enumerate(sorted(int(x) for x in values))}

def make_sequences(df, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab):
    sequences, labels = [], []
    for _, g in df.groupby("window_id", sort=False):
        g = g.reset_index(drop=True)
        y = g["label"].to_numpy(dtype=np.int64)
        label = int(np.bincount(y, minlength=len(NODE1_CLASSES)).argmax())
        num = g[numeric_cols].to_numpy(dtype=np.float32)
        num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0)
        num = scaler.transform(num).astype(np.float32)
        src_raw = pd.to_numeric(g["Source Port"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        dst_raw = pd.to_numeric(g["Destination Port"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        proto_raw = pd.to_numeric(g["Protocol"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        src = np.array([src_vocab.get(int(v), 0) for v in src_raw], dtype=np.int64)
        dst = np.array([dst_vocab.get(int(v), 0) for v in dst_raw], dtype=np.int64)
        proto = np.array([proto_vocab.get(int(v), 0) for v in proto_raw], dtype=np.int64)
        sequences.append((num, src, dst, proto)); labels.append(label)
    return sequences, np.asarray(labels, dtype=np.int64)

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return (*self.sequences[idx], int(self.labels[idx]))

class SequenceCollator:
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
                torch.from_numpy(proto), torch.tensor(lengths, dtype=torch.long), torch.from_numpy(y))

class LSTMClassifier(nn.Module):
    def __init__(self, n_numeric, src_cardinality, dst_cardinality, protocol_cardinality,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, bidirectional=BIDIRECTIONAL):
        super().__init__()
        self.src_embedding = nn.Embedding(src_cardinality, PORT_EMBED_DIM)
        self.dst_embedding = nn.Embedding(dst_cardinality, PORT_EMBED_DIM)
        self.protocol_embedding = nn.Embedding(protocol_cardinality, PROTOCOL_EMBED_DIM)
        input_size = n_numeric + PORT_EMBED_DIM * 2 + PROTOCOL_EMBED_DIM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_size)
        self.head = nn.Sequential(nn.Linear(out_size, hidden_size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_size, len(NODE1_CLASSES)))
    def forward(self, num, src, dst, protocol, lengths):
        x = torch.cat([num, self.src_embedding(src), self.dst_embedding(dst), self.protocol_embedding(protocol)], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.lstm.bidirectional else h_n[-1]
        return self.head(self.norm(h))

class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, weight=None):
        super().__init__()
        self.gamma = gamma
        if weight is not None: self.register_buffer("weight", weight)
        else: self.weight = None
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum().clamp_min(1e-8)
        return loss.mean()

def evaluate_loader(model, loader, device):
    model.eval()
    correct = 0; total = 0
    preds, targets = [], []
    with torch.no_grad():
        for num, src, dst, proto, lengths, y in loader:
            num = num.to(device); src = src.to(device); dst = dst.to(device)
            proto = proto.to(device); lengths = lengths.to(device); y = y.to(device)
            logits = model(num, src, dst, proto, lengths)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item(); total += len(y)
            preds.append(pred.cpu().numpy()); targets.append(y.cpu().numpy())
    return correct / max(total, 1), np.concatenate(preds), np.concatenate(targets)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    with open(PROCESSED_DIR / "feature_names.json", encoding="utf-8") as f:
        features = json.load(f)
    numeric_cols = [c for c in features if c not in FORCED_CATEGORICAL]

    train_df = load_split("train"); val_df = load_split("val"); test_df = load_split("test")
    
    scaler = StandardScaler()
    train_num = train_df[numeric_cols].to_numpy(dtype=np.float32)
    scaler.fit(np.nan_to_num(train_num, nan=0.0, posinf=0.0, neginf=0.0))

    src_vocab = build_vocab(train_df["Source Port"])
    dst_vocab = build_vocab(train_df["Destination Port"])
    proto_vocab = build_vocab(train_df["Protocol"])

    train_seq, y_train = make_sequences(train_df, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab)
    val_seq, y_val = make_sequences(val_df, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab)
    test_seq, y_test = make_sequences(test_df, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab)

    class_counts = np.bincount(y_train, minlength=len(NODE1_CLASSES)).astype(np.float64)
    sample_weights = (1.0 / np.maximum(class_counts, 1.0))[y_train]
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(y_train), replacement=True)

    train_loader = DataLoader(SequenceDataset(train_seq, y_train), batch_size=BATCH_SIZE, sampler=sampler, collate_fn=SequenceCollator())
    val_loader = DataLoader(SequenceDataset(val_seq, y_val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=SequenceCollator())
    test_loader = DataLoader(SequenceDataset(test_seq, y_test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=SequenceCollator())

    weights = torch.tensor(balanced_weights(y_train, len(NODE1_CLASSES)), dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=weights)

    model = LSTMClassifier(n_numeric=len(numeric_cols), src_cardinality=len(src_vocab) + 1,
                           dst_cardinality=len(dst_vocab) + 1, protocol_cardinality=len(proto_vocab) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0; train_count = 0
        epoch_start = time.perf_counter()
        
        for num, src, dst, proto, lengths, y in train_loader:
            num = num.to(device); src = src.to(device); dst = dst.to(device)
            proto = proto.to(device); lengths = lengths.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(num, src, dst, proto, lengths)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(y); train_count += len(y)

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc, _, _ = evaluate_loader(model, train_loader, device)
        elapsed = time.perf_counter() - epoch_start

        _, val_pred, val_true = evaluate_loader(model, val_loader, device)
        val_f1 = float(f1_score(val_true, val_pred, labels=list(range(len(NODE1_CLASSES))), average="macro", zero_division=0))

        history.append({"epoch": epoch, "train_loss": round(float(train_loss), 6), "train_accuracy": round(float(train_acc), 6)})
        print(f"Эпоха {epoch:>2}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} ({elapsed:.1f} с)")

        if val_f1 > best_f1 + 1e-5:
            best_f1, best_state, wait = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping на {epoch}-й эпохе (скрыто).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(), "classes": NODE1_CLASSES, "classes_ru": CLASS_NAMES_RU,
        "features": features, "numeric_cols": numeric_cols, "categorical_cols": FORCED_CATEGORICAL,
        "src_vocab": src_vocab, "dst_vocab": dst_vocab, "protocol_vocab": proto_vocab,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "config": {"model": "BiLSTM", "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
                   "dropout": DROPOUT, "max_flows_per_window": MAX_FLOWS_PER_WINDOW, "focal_gamma": FOCAL_GAMMA},
    }, MODEL_DIR / "model.pt")

    _, y_pred, y_true = evaluate_loader(model, test_loader, device)
    metrics = evaluate(y_true, y_pred, CLASS_NAMES_RU)
    
    result = {"node": 1, "model": "BiLSTM", "classes": NODE1_CLASSES, "classes_ru": CLASS_NAMES_RU,
              "macro_f1": metrics["f1_macro"], "accuracy": metrics["accuracy"], "best_val_macro_f1": best_f1}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (MODEL_DIR / "training_history.json").write_text(json.dumps({"history": history}, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_confusion_matrices(metrics["confusion_matrix"], CLASS_NAMES_RU, MODEL_DIR)
    plot_training_curves(history, MODEL_DIR)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])

if __name__ == "__main__":
    main()