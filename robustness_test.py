import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Импортируем твои классы и функции
from labels import NODE1_CLASSES
from window_features import aggregate_windows, feature_columns
# Для LSTM
from node1_lstm_1 import LSTMClassifier, build_vocab, make_sequences, SequenceDataset, SequenceCollator
from torch.utils.data import DataLoader

SEED = 42
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_DIR = Path("robustness_results")
OUT_DIR.mkdir(exist_ok=True)

def add_noise(X, noise_level, numeric_indices):
    """Добавляет гауссов шум к числовым признакам"""
    if noise_level == 0:
        return X
    X_noisy = X.copy()
    # Считаем стандартное отклонение для каждого признака
    stds = np.std(X, axis=0)
    # Добавляем шум только к числовым колонкам (не трогаем порты/протоколы)
    for idx in numeric_indices:
        noise = np.random.normal(0, noise_level * stds[idx], size=X.shape[0])
        X_noisy[:, idx] += noise
    return X_noisy

def test_lstm_noise(noise_levels):
    print("\n--- Тестирование LSTM на устойчивость к шуму ---")
    device = torch.device("cpu")
    ckpt = torch.load(MODEL_DIR / "node1_lstm" / "model.pt", map_location=device, weights_only=False)
    
    numeric_cols = ckpt["numeric_cols"]
    src_vocab = {int(k): int(v) for k, v in ckpt["src_vocab"].items()}
    dst_vocab = {int(k): int(v) for k, v in ckpt["dst_vocab"].items()}
    proto_vocab = {int(k): int(v) for k, v in ckpt["protocol_vocab"].items()}
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"])
    scaler.scale_ = np.array(ckpt["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(numeric_cols)

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    
    model = LSTMClassifier(n_numeric=len(numeric_cols),
                           src_cardinality=len(src_vocab) + 1,
                           dst_cardinality=len(dst_vocab) + 1,
                           protocol_cardinality=len(proto_vocab) + 1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    results = []
    for noise in noise_levels:
        # Временно подменяем данные с шумом (делаем копию)
        df_noisy = test_df.copy()
        num_data = df_noisy[numeric_cols].to_numpy(dtype=np.float32)
        num_data = np.nan_to_num(num_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Добавляем шум
        stds = np.std(num_data, axis=0)
        for i in range(num_data.shape[1]):
            noise_arr = np.random.normal(0, noise * stds[i], size=num_data.shape[0])
            num_data[:, i] += noise_arr
        df_noisy[numeric_cols] = num_data

        # Готовим последовательности
        seqs, y_true = make_sequences(df_noisy, numeric_cols, scaler, src_vocab, dst_vocab, proto_vocab)
        loader = DataLoader(SequenceDataset(seqs, y_true), batch_size=64, shuffle=False, collate_fn=SequenceCollator())
        
        preds = []
        with torch.no_grad():
            for num, src, dst, proto, lengths, y in loader:
                logits = model(num, src, dst, proto, lengths)
                preds.append(logits.argmax(dim=1).numpy())
        y_pred = np.concatenate(preds)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=list(range(len(NODE1_CLASSES))), average="macro", zero_division=0)
        results.append((noise, acc, f1))
        print(f"Шум {noise*100:>4.1f}%: Accuracy = {acc:.4f}, Macro-F1 = {f1:.4f}")
    
    return results

def test_lgbm_noise(noise_levels):
    print("\n--- Тестирование LightGBM на устойчивость к шуму ---")
    import lightgbm as lgb
    with open(MODEL_DIR / "node1_lgbm" / "feature_columns.json") as f:
        feat_cols = json.load(f)
    with open(PROCESSED_DIR / "feature_names.json") as f:
        features = json.load(f)
        
    numeric_cols = [c for c in features if c not in ["Source Port", "Destination Port", "Protocol"]]
    categorical_cols = ["Source Port", "Destination Port", "Protocol"]
    
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    booster = lgb.Booster(model_file=str(MODEL_DIR / "node1_lgbm" / "model.txt"))
    
    results = []
    for noise in noise_levels:
        df_noisy = test_df.copy()
        num_data = df_noisy[numeric_cols].to_numpy(dtype=np.float32)
        num_data = np.nan_to_num(num_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        stds = np.std(num_data, axis=0)
        for i in range(num_data.shape[1]):
            noise_arr = np.random.normal(0, noise * stds[i], size=num_data.shape[0])
            num_data[:, i] += noise_arr
        df_noisy[numeric_cols] = num_data

        agg = aggregate_windows(df_noisy, numeric_cols, categorical_cols)
        X = agg[feat_cols].to_numpy()
        y_true = agg["label"].to_numpy()
        
        y_pred = booster.predict(X).argmax(axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=list(range(len(NODE1_CLASSES))), average="macro", zero_division=0)
        results.append((noise, acc, f1))
        print(f"Шум {noise*100:>4.1f}%: Accuracy = {acc:.4f}, Macro-F1 = {f1:.4f}")
        
    return results

def plot_results(lstm_results, lgbm_results):
    plt.figure(figsize=(10, 6))
    
    lstm_noise = [r[0]*100 for r in lstm_results]
    lstm_f1 = [r[2] for r in lstm_results]
    lgbm_noise = [r[0]*100 for r in lgbm_results]
    lgbm_f1 = [r[2] for r in lgbm_results]
    
    plt.plot(lstm_noise, lstm_f1, marker='o', label="LSTM", linewidth=2)
    plt.plot(lgbm_noise, lgbm_f1, marker='s', label="LightGBM", linewidth=2)
    
    plt.xlabel("Уровень шума (%)")
    plt.ylabel("Macro-F1")
    plt.title("Оценка робастности моделей (ГОСТ Р 59898-2021)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(OUT_DIR / "robustness_curve.png", dpi=150)
    print(f"\nГрафик сохранен в {OUT_DIR}/robustness_curve.png")

if __name__ == "__main__":
    np.random.seed(SEED)
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
    
    lstm_res = test_lstm_noise(noise_levels)
    lgbm_res = test_lgbm_noise(noise_levels)
    plot_results(lstm_res, lgbm_res)