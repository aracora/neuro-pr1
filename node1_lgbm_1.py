#!/usr/bin/env python
"""Узел 1 — LightGBM (Бинарная классификация: BENIGN vs ATTACK)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from labels_node1 import NODE1_CLASSES, CLASS_LABELS_RU, CLASS_NAMES_RU
from window_features import aggregate_windows, feature_columns

SEED = 42
PROCESSED_DIR = Path("data/processed_node1")
MODEL_DIR = Path("models/node1_lgbm_1")

FORCED_CATEGORICAL = ["Source Port", "Destination Port", "Protocol"]
LGBM_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt", 
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
    "class_weight": "balanced", "random_state": SEED, "n_jobs": -1, "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 30

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=classes_ru, yticklabels=classes_ru, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок — Узел 1 (LightGBM)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / "node1_lgbm_1_confusion_matrix.png", dpi=150); plt.close()

    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges", xticklabels=classes_ru, yticklabels=classes_ru, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title("Нормализованная матрица ошибок — Узел 1 (LightGBM)")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out_dir / "node1_lgbm_1_confusion_matrix_normalized.png", dpi=150); plt.close()

def main():
    import lightgbm as lgb
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric=["binary_logloss", "binary_error"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True)]
    )

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, CLASS_NAMES_RU)
    
    val_pred = model.predict(X_val)
    val_f1 = float(f1_score(y_val, val_pred, labels=list(range(len(NODE1_CLASSES))), average="macro", zero_division=0))

    model.booster_.save_model(str(MODEL_DIR / "model.txt"))
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2, ensure_ascii=False)

    result = {"node": 1, "model": "LightGBM", "classes": NODE1_CLASSES, "classes_ru": CLASS_NAMES_RU,
              "macro_f1": metrics["f1_macro"], "accuracy": metrics["accuracy"], "best_val_macro_f1": val_f1}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_confusion_matrices(metrics["confusion_matrix"], CLASS_NAMES_RU, MODEL_DIR)
    
    res = model.evals_result_
    iters = np.arange(1, len(res["train"]["binary_logloss"]) + 1)
    plt.figure(figsize=(9, 6))
    plt.plot(iters, res["train"]["binary_logloss"], label="Обучающая выборка")
    plt.xlabel("Итерация"); plt.ylabel("binary_logloss")
    plt.title("Процесс обучения Узла 1 — LightGBM (потери)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(MODEL_DIR / "node1_lgbm_1_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(9, 6))
    acc = 1.0 - np.asarray(res["train"]["binary_error"], dtype=float)
    plt.plot(iters, acc, label="Обучающая выборка")
    plt.xlabel("Итерация"); plt.ylabel("Точность")
    plt.title("Процесс обучения Узла 1 — LightGBM (точность)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(MODEL_DIR / "node1_lgbm_1_accuracy.png", dpi=150); plt.close()

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])

if __name__ == "__main__":
    main()