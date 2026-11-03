#!/usr/bin/env python
"""Двухузловой пайплайн: Узел 1 (Бинарный) + Узел 2 (Мультиклассовый)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Импорты из наших скриптов
from labels_node1 import CLASS_NAMES_RU as NAMES_1, NODE1_CLASSES as CLS_1
from labels_node2 import CLASS_NAMES_RU as NAMES_2, NODE1_CLASSES as CLS_2
from window_features import aggregate_windows, feature_columns

# Для Узла 1
from node1_lstm_1 import LSTMClassifier as LSTM1, build_vocab, make_sequences as make_seq1, SequenceDataset as SeqDs1, SequenceCollator as SeqColl1
from node1_mlp_1 import MLPClassifier as MLP1

# Для Узла 2
from node1_lstm_2 import LSTMClassifier as LSTM2, make_sequences as make_seq2, SequenceDataset as SeqDs2, SequenceCollator as SeqColl2
from node1_mlp_2 import MLPClassifier as MLP2

import lightgbm as lgb

SEED = 42
PROCESSED_1 = Path("data/processed_node1")
PROCESSED_2 = Path("data/processed_node2")
MODELS = Path("models")
OUT_DIR = Path("pipeline_results")
OUT_DIR.mkdir(exist_ok=True)

def get_lstm_probs(ckpt_dir, test_df, LSTMClass, make_seq_fn, SeqDs, SeqColl):
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=False)
    numeric_cols = ckpt["numeric_cols"]
    src_vocab = {int(k): int(v) for k, v in ckpt["src_vocab"].items()}
    dst_vocab = {int(k): int(v) for k, v in ckpt["dst_vocab"].items()}
    proto_vocab = {int(k): int(v) for k, v in ckpt["protocol_vocab"].items()}

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"])
    scaler.scale_ = np.array(ckpt["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(numeric_cols)

    seqs, labels = make_seq_fn(test_df, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab)
    model = LSTMClass(n_numeric=len(numeric_cols), src_cardinality=len(src_vocab) + 1,
                      dst_cardinality=len(dst_vocab) + 1, protocol_cardinality=len(proto_vocab) + 1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(SeqDs(seqs, labels), batch_size=64, shuffle=False, collate_fn=SeqColl())
    probs_list = []
    with torch.no_grad():
        for num, src, dst, proto, lengths, y in loader:
            logits = model(num, src, dst, proto, lengths)
            probs_list.append(F.softmax(logits, dim=-1).numpy())
    return np.concatenate(probs_list)

def get_mlp_probs(ckpt_dir, agg_df, MLPClass, feat_cols):
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=False)
    X = agg_df[feat_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"])
    scaler.scale_ = np.array(ckpt["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = X.shape[1]
    X = scaler.transform(X).astype(np.float32)

    model = MLPClass(n_features=X.shape[1], n_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    with torch.no_grad():
        probs = F.softmax(model(torch.from_numpy(X).to(device)), dim=-1).numpy()
    return probs

def main():
    print("=== ЗАПУСК ДВУХУЗЛОВОГО ПАЙПЛАЙНА ===")
    
    # 1. Загружаем данные
    print("Агрегация тестовых окон...")
    test_df_1 = pd.read_parquet(PROCESSED_1 / "test.parquet")
    test_df_2 = pd.read_parquet(PROCESSED_2 / "test.parquet")
    
    with open(PROCESSED_1 / "feature_names.json") as f:
        features = json.load(f)
    num_cols = [c for c in features if c not in ["Source Port", "Destination Port", "Protocol"]]
    cat_cols = ["Source Port", "Destination Port", "Protocol"]
    
    # Агрегируем оба набора
    agg_1 = aggregate_windows(test_df_1, num_cols, cat_cols).sort_values("window_id").reset_index(drop=True)
    agg_2 = aggregate_windows(test_df_2, num_cols, cat_cols).sort_values("window_id").reset_index(drop=True)
    
    # Синхронизируем по window_id
    common_windows = set(agg_1["window_id"]).intersection(set(agg_2["window_id"]))
    agg_1 = agg_1[agg_1["window_id"].isin(common_windows)].reset_index(drop=True)
    agg_2 = agg_2.set_index("window_id").loc[agg_1["window_id"]].reset_index()

    with open(MODELS / "node1_lgbm_1" / "feature_columns.json") as f:
        feat_cols = json.load(f)

    # ---------------- УЗЕЛ 1 ----------------
    print("\n[Узел 1] Предсказание: Норма или Атака...")
    
    p_mlp_1 = get_mlp_probs(MODELS / "node1_mlp_1", agg_1, MLP1, feat_cols)
    
    X_lgbm = agg_1[feat_cols].to_numpy(dtype=np.float32)
    X_lgbm = np.nan_to_num(X_lgbm, nan=0.0, posinf=0.0, neginf=0.0)
    booster_1 = lgb.Booster(model_file=str(MODELS / "node1_lgbm_1" / "model.txt"))
    p_lgbm_1 = booster_1.predict(X_lgbm)
    if p_lgbm_1.ndim == 1:
        p_lgbm_1 = np.stack([1 - p_lgbm_1, p_lgbm_1], axis=1)

    # Ансамбль Узла 1 (MLP + LGBM)
    p_ensemble_1 = (p_mlp_1 + p_lgbm_1) / 2.0
    pred_1 = p_ensemble_1.argmax(axis=1)
    
    y_true_1 = agg_1["label"].to_numpy()
    
    print(f"Узел 1 (MLP + LGBM) Accuracy: {accuracy_score(y_true_1, pred_1):.4f}")
    print(f"Узел 1 Macro-F1: {f1_score(y_true_1, pred_1, average='macro'):.4f}")

    # ---------------- УЗЕЛ 2 ----------------
    print("\n[Узел 2] Предсказание типа атаки (для тех, кого Узел 1 назвал атакой)...")
    # Находим индексы окон, которые Узел 1 назвал АТАКОЙ (класс 1)
    attack_indices = np.where(pred_1 == 1)[0]
    
    if len(attack_indices) == 0:
        print("Узел 1 не нашел атак!")
        return

    # Берем эти же окна из Узла 2
    agg_2_attacks = agg_2.iloc[attack_indices].reset_index(drop=True)
    y_true_2 = agg_2_attacks["label"].to_numpy()

    p_mlp_2 = get_mlp_probs(MODELS / "node1_mlp_2", agg_2_attacks, MLP2, feat_cols)
    
    X_lgbm_2 = agg_2_attacks[feat_cols].to_numpy(dtype=np.float32)
    X_lgbm_2 = np.nan_to_num(X_lgbm_2, nan=0.0, posinf=0.0, neginf=0.0)
    booster_2 = lgb.Booster(model_file=str(MODELS / "node1_lgbm_2" / "model.txt"))
    p_lgbm_2 = booster_2.predict(X_lgbm_2)

    # Ансамбль Узла 2 (MLP + LGBM)
    p_ensemble_2 = (p_mlp_2 + p_lgbm_2) / 2.0
    pred_2 = p_ensemble_2.argmax(axis=1)
    
    print(f"Узел 2 (MLP + LGBM) Accuracy: {accuracy_score(y_true_2, pred_2):.4f}")
    print(f"Узел 2 Macro-F1: {f1_score(y_true_2, pred_2, average='macro'):.4f}")

    # ---------------- ОБЩИЙ РЕЗУЛЬТАТ ----------------
    final_preds = ["Нормальный трафик"] * len(y_true_1)
    for i, attack_idx in enumerate(attack_indices):
        final_preds[attack_idx] = NAMES_2[pred_2[i]]

    final_classes = ["Нормальный трафик"] + NAMES_2
    y_true_final = []
    for i in range(len(y_true_1)):
        if y_true_1[i] == 0: # BENIGN
            y_true_final.append("Нормальный трафик")
        else:
            # Берем истинную метку из Узла 2 для этого индекса
            y_true_final.append(NAMES_2[y_true_2[np.where(attack_indices == i)[0][0]]])

    print("\n=== ИТОГОВЫЙ РЕЗУЛЬТАТ ПАЙПЛАЙНА ===")
    print(classification_report(y_true_final, final_preds, labels=final_classes, zero_division=0))

    # Рисуем итоговую матрицу ошибок
    cm = confusion_matrix(y_true_final, final_preds, labels=final_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=final_classes, yticklabels=final_classes, cbar=False)
    plt.xlabel("Предсказанный класс"); plt.ylabel("Истинный класс")
    plt.title("Итоговая матрица ошибок пайплайна")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(OUT_DIR / "pipeline_confusion_matrix.png", dpi=150)
    print(f"\nГрафик сохранен в {OUT_DIR}/pipeline_confusion_matrix.png")

if __name__ == "__main__":
    main()