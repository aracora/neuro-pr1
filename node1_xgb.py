#!/usr/bin/env python
"""Узел 1 — XGBoost-классификация сетевого трафика по временным окнам.

Использует ту же агрегацию окна, что и node1_mlp.py / node1_lgbm.py:
  * числовые (77): mean, std, min, max -> 308
  * категориальные (3): mode -> 3
  Итого 311 признаков.

Запуск:
    python preprocess.py --raw data/raw --out data/processed --window 5
    python node1_xgb.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from labels import NODE1_CLASSES
from window_features import aggregate_windows, feature_columns

SEED = 42
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/node1_xgb")

FORCED_CATEGORICAL = ["Source Port", "Destination Port", "Protocol"]

EARLY_STOPPING_ROUNDS = 30
LOG_PERIOD = 50

# Гиперпараметры XGBoost (аналогичны node1_lgbm.py).
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(NODE1_CLASSES),
    "booster": "gbtree",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "tree_method": "hist",
    "eval_metric": ["mlogloss", "merror"],
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "random_state": SEED,
    "n_jobs": -1,
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    n_classes = len(classes)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=classes, zero_division=0
        ),
        "confusion_matrix": cm,
    }


def plot_confusion_matrices(cm: np.ndarray, classes: list[str], out_dir: Path) -> None:
    size = max(6, len(classes) * 1.2)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок — XGBoost")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "node1_xgb_confusion_matrix.png", dpi=150)
    plt.close()

    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), denom,
                        out=np.zeros_like(cm, dtype=float), where=denom != 0)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Нормализованная матрица ошибок — XGBoost")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "node1_xgb_confusion_matrix_normalized.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feat_cols: list[str], out_dir: Path, top_k: int = 30) -> None:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_k]
    names = [feat_cols[i] for i in order]
    values = imp[order]

    plt.figure(figsize=(9, max(6, top_k * 0.3)))
    plt.barh(range(len(names)), values[::-1])
    plt.yticks(range(len(names)), names[::-1], fontsize=8)
    plt.xlabel("Важность признака")
    plt.title(f"XGBoost — топ-{top_k} признаков по важности")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "node1_xgb_feature_importance.png", dpi=150)
    plt.close()


def plot_training_curves(model, out_dir: Path) -> dict:
    """Графики loss и accuracy по итерациям — в стиле LSTM."""
    res = model.evals_result()
    names = list(res.keys())
    if not names:
        print("[!] evals_result пуст, графики обучения пропущены.")
        return {}

    n_iter = len(res[names[0]]["mlogloss"])
    iters = np.arange(1, n_iter + 1)
    best_it = int(getattr(model, "best_iteration", 0) or 0)

    pretty = {
        "validation_0": "Обучающая выборка",
        "validation_1": "Валидационная выборка",
    }

    # --- Loss ---
    plt.figure(figsize=(9, 6))
    for name in names:
        plt.plot(iters, res[name]["mlogloss"], label=pretty.get(name, name))
    plt.xlabel("Итерация бустинга")
    plt.ylabel("Значение функции потерь (mlogloss)")
    plt.title("Процесс обучения XGBoost — функция потерь")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "node1_xgb_loss.png", dpi=150)
    plt.close()

    # --- Accuracy ---
    plt.figure(figsize=(9, 6))
    for name in names:
        acc = 1.0 - np.asarray(res[name]["merror"], dtype=float)
        plt.plot(iters, acc, label=pretty.get(name, name))
    plt.xlabel("Итерация бустинга")
    plt.ylabel("Точность")
    plt.title("Процесс обучения XGBoost — точность")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "node1_xgb_accuracy.png", dpi=150)
    plt.close()

    history = {
        "n_iterations": int(n_iter),
        "best_iteration": best_it,
        "per_iteration": {},
    }
    for name in names:
        history["per_iteration"][name] = {
            "mlogloss": [float(x) for x in res[name]["mlogloss"]],
            "accuracy": [float(1.0 - x) for x in res[name]["merror"]],
        }
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Узел 1 — XGBoost по временным окнам")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Оставлено для совместимости")
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("XGBoost не установлен. Выполни: pip install xgboost")

    with open(PROCESSED_DIR / "feature_names.json", encoding="utf-8") as f:
        features = json.load(f)

    numeric_cols = [c for c in features if c not in FORCED_CATEGORICAL]
    categorical_cols = list(FORCED_CATEGORICAL)

    if len(features) != 80:
        raise SystemExit(f"Ожидалось 80 признаков, получено {len(features)}")

    print("Агрегация окон...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    train_agg = aggregate_windows(train_df, numeric_cols, categorical_cols)
    val_agg = aggregate_windows(val_df, numeric_cols, categorical_cols)
    test_agg = aggregate_windows(test_df, numeric_cols, categorical_cols)

    feat_cols = feature_columns(numeric_cols, categorical_cols)
    print(f"Агрегированных признаков: {len(feat_cols)}")

    X_train = train_agg[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_agg["label"].to_numpy(dtype=np.int64)
    X_val = val_agg[feat_cols].to_numpy(dtype=np.float32)
    y_val = val_agg["label"].to_numpy(dtype=np.int64)
    X_test = test_agg[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_agg["label"].to_numpy(dtype=np.int64)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Окон: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Классы в train: {[NODE1_CLASSES[i] for i in sorted(np.unique(y_train))]}")
    class_counts = np.bincount(y_train, minlength=len(NODE1_CLASSES))
    print(f"Окон в train по классам: "
          f"{ {NODE1_CLASSES[i]: int(class_counts[i]) for i in range(len(NODE1_CLASSES))} }")

    if len(np.unique(y_train)) < 2:
        raise SystemExit("В train < 2 классов — обучение бессмысленно.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Балансировка классов (аналог class_weight="balanced" у LightGBM).
    sample_weights = compute_sample_weight("balanced", y_train)
    print(f"Диапазон sample_weight: "
          f"[{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    print(f"XGBoost: n_estimators={XGB_PARAMS['n_estimators']}, "
          f"lr={XGB_PARAMS['learning_rate']}, max_depth={XGB_PARAMS['max_depth']}")

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    best_it = int(getattr(model, "best_iteration", XGB_PARAMS["n_estimators"] - 1))
    print(f"Лучшая итерация: {best_it}")

    val_pred = model.predict(X_val)
    val_f1 = float(f1_score(y_val, val_pred,
                            labels=list(range(len(NODE1_CLASSES))),
                            average="macro", zero_division=0))
    print(f"Val macro-F1: {val_f1:.4f}")

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, NODE1_CLASSES)

    model.save_model(str(MODEL_DIR / "model.json"))
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2, ensure_ascii=False)

    result = {
        "node": 1,
        "model": "XGBoost",
        "classes": NODE1_CLASSES,
        "n_classes": len(NODE1_CLASSES),
        "n_train_windows": int(len(y_train)),
        "n_val_windows": int(len(y_val)),
        "n_test_windows": int(len(y_test)),
        "n_features": len(feat_cols),
        "best_iteration": best_it,
        "best_val_macro_f1": round(val_f1, 6),
        "macro_f1": metrics["f1_macro"],
        "accuracy": metrics["accuracy"],
        "seed": SEED,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_confusion_matrices(metrics["confusion_matrix"], NODE1_CLASSES, MODEL_DIR)
    plot_feature_importance(model, feat_cols, MODEL_DIR)
    history = plot_training_curves(model, MODEL_DIR)
    if history:
        (MODEL_DIR / "training_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])
    print(f"Артефакты сохранены в {MODEL_DIR}/")


if __name__ == "__main__":
    main()