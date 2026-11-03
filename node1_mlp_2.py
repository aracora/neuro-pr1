#!/usr/bin/env python
"""Узел 2 — MLP (Мультиклассовая классификация типов атак)."""
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from labels_node2 import NODE1_CLASSES, CLASS_LABELS_RU, CLASS_NAMES_RU
from window_features import aggregate_windows, feature_columns

SEED = 42
PROCESSED_DIR = Path("data/processed_node2")
MODEL_DIR = Path("models/node1_mlp_2")

BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_SIZES = (256, 128)
DROPOUT = 0.3
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes_ru, yticklabels=classes_ru, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок — Узел 2 (MLP)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / "node1_mlp_2_confusion_matrix.png", dpi=150); plt.close()

    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=classes_ru, yticklabels=classes_ru, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title("Нормализованная матрица ошибок — Узел 2 (MLP)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / "node1_mlp_2_confusion_matrix_normalized.png", dpi=150); plt.close()

def plot_training_curves(history, out_dir: Path):
    epochs = [e["epoch"] for e in history]
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [e["train_loss"] for e in history], label="Focal loss (train)")
    plt.xlabel("Эпоха"); plt.ylabel("Loss")
    plt.title("Процесс обучения Узла 2 — MLP (потери)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "node1_mlp_2_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [e["train_accuracy"] for e in history], label="Точность (train)")
    plt.xlabel("Эпоха"); plt.ylabel("Точность")
    plt.title("Процесс обучения Узла 2 — MLP (точность)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "node1_mlp_2_accuracy.png", dpi=150); plt.close()

class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLPClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_sizes=HIDDEN_SIZES, dropout=DROPOUT):
        super().__init__()
        layers = []; prev = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

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
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
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
    categorical_cols = list(FORCED_CATEGORICAL)

    print("Агрегация окон...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    
    train_agg = aggregate_windows(train_df, numeric_cols, categorical_cols)
    val_agg = aggregate_windows(val_df, numeric_cols, categorical_cols)
    test_agg = aggregate_windows(test_df, numeric_cols, categorical_cols)

    feat_cols = feature_columns(numeric_cols, categorical_cols)
    X_train = train_agg[feat_cols].to_numpy(dtype=np.float32); y_train = train_agg["label"].to_numpy(dtype=np.int64)
    X_val = val_agg[feat_cols].to_numpy(dtype=np.float32); y_val = val_agg["label"].to_numpy(dtype=np.int64)
    X_test = test_agg[feat_cols].to_numpy(dtype=np.float32); y_test = test_agg["label"].to_numpy(dtype=np.int64)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    class_counts = np.bincount(y_train, minlength=len(NODE1_CLASSES)).astype(np.float64)
    sample_weights = (1.0 / np.maximum(class_counts, 1.0))[y_train]
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(y_train), replacement=True)

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    weights = torch.tensor(balanced_weights(y_train, len(NODE1_CLASSES)), dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=weights)

    model = MLPClassifier(n_features=X_train.shape[1], n_classes=len(NODE1_CLASSES)).to(device)
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
        
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
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
        print(f"Эпоха {epoch:>3}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} ({elapsed:.1f} с)")

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
        "feature_columns": feat_cols, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "config": {"model": "MLP", "hidden_sizes": list(HIDDEN_SIZES), "dropout": DROPOUT, "focal_gamma": FOCAL_GAMMA},
    }, MODEL_DIR / "model.pt")

    _, y_pred, y_true = evaluate_loader(model, test_loader, device)
    metrics = evaluate(y_true, y_pred, CLASS_NAMES_RU)

    result = {"node": 2, "model": "MLP", "classes": NODE1_CLASSES, "classes_ru": CLASS_NAMES_RU,
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