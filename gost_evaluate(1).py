#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оценка узлов обнаружения DDoS по формулам 13–17 ГОСТ Р 59898-2021.

Применение (из корня проекта):
    python gost_evaluate.py
    python gost_evaluate.py --skip-lstm-inference

Оценка выполнена по исходным flow тестовой выборки. Для оконных моделей
предсказание окна распространяется на все входящие в него потоки.

Автоматически находит models/**/metrics.json независимо от имен каталогов
(например, models/node_lgbm1/metrics.json) и pipeline_results/metrics.json. Если у ранее обученной BiLSTM нет counts в JSON,
восстанавливает их на тестовом Parquet без переобучения, используя model.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

FORMULAS = {
    "accuracy": "(13) (TP + TN) / (TP + TN + FP + FN)",
    "precision": "(14) TP / (TP + FP)",
    "recall": "(15) TP / (TP + FN)",
    "specificity": "(16) TN / (TN + FP)",
    "f1": "(17) 2 * Precision * Recall / (Precision + Recall)",
}


def _divide(a, b):
    """Возвращает None, если по формуле деление на ноль."""
    return float(a / b) if b else None


def _as_display(v):
    return "—" if v is None else f"{v:.4f}"


def _avg(values):
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def evaluate_confusion_matrix(matrix, class_names):
    """Метрики ГОСТ для многоклассовой задачи: one-vs-rest для каждого класса.

    Rows = реальные классы, columns = предсказанные. Взаимоисключающие
    TP/TN/FP/FN учитываются по количеству flow, не по нормализованным долям.
    """
    cm = np.asarray(matrix)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Нужна квадратная confusion_matrix_counts")
    if not len(class_names) == cm.shape[0]:
        raise ValueError(f"Ожидалось классов {len(class_names)}, получено {cm.shape[0]}")
    if not np.all(np.isfinite(cm)) or np.any(cm < 0) or np.any(cm != np.floor(cm)):
        raise ValueError("Матрица должна содержать неотрицательные целые количества flow")
    cm = cm.astype(np.int64)
    total = int(cm.sum())
    if total < 1:
        raise ValueError("Пустая матрица: данные тестовой выборки отсутствуют")
    per_class = []
    for cls, name in enumerate(class_names):
        tp = int(cm[cls, cls])
        fn = int(cm[cls, :].sum()) - tp
        fp = int(cm[:, cls].sum()) - tp
        tn = total - tp - fn - fp
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        specificity = _divide(tn, tn + fp)
        f1 = (_divide(2 * precision * recall, precision + recall)
              if precision is not None and recall is not None else None)
        # Для каждой задачи one-vs-rest формула (13) применяется буквально.
        acc = _divide(tp + tn, total)
        per_class.append({
            "class": str(name), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "support": tp + fn,
            "accuracy": acc, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1,
        })

    # Multiclass accuracy = сумма диагонали / общее число flow.
    # Не усредняем one-vs-rest accuracy, т.к. TN там посчитан повторно.
    overall_accuracy = float(np.trace(cm) / total)
    macro = {f"{m}_macro": _avg([r[m] for r in per_class])
             for m in ("precision", "recall", "specificity", "f1")}
    weights = np.array([r["support"] for r in per_class], dtype=float)
    f1_vals = [r["f1"] for r in per_class]
    f1_weighted = float(sum((f if f is not None else 0) * w
                            for f, w in zip(f1_vals, weights)) / total)
    out = {
        "standard": "ГОСТ Р 59898-2021, таблица 5, формулы (13)–(17)",
        "evaluation_unit": "network_flow",
        "test_flows": total,
        "correct_flows": int(np.trace(cm)),
        "classes": list(class_names),
        "accuracy": overall_accuracy,
        **macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "confusion_matrix_counts": cm.tolist(),
        "formulas": FORMULAS,
        "note": ("При отсутствии соответствующих положительных или отрицательных "
                 "наблюдений метрика с нулевым знаменателем не определена "
                 "и записана как null; macro считается по определённым значениям. "
                 "Для многоклассовой задачи TP/TN/FP/FN вычисляются one-vs-rest."),
    }
    return out


def save_reports(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gost_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "gost_metrics_by_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "tp", "tn", "fp", "fn", "support",
                                               "accuracy", "precision", "recall", "specificity", "f1"])
        writer.writeheader()
        writer.writerows(result["per_class"])


def _lstm_counts(node, project_root, device="auto", batch_size=64, model_dir=None):
    """Получить counts по сохранённой LSTM без переподготовки / переобучения.

    Загружает только модель и тестовый Parquet потоково. Код совместим с
    ранее предоставленными node1_lstm_1.py и node1_lstm_2.py.
    """
    import importlib
    import torch
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm

    module = importlib.import_module(f"node1_lstm_{node}")
    ckpt_path = (model_dir or (project_root / "models" / f"node1_lstm_{node}")) / "model.pt"
    parquet_path = project_root / "data" / f"processed_node{node}" / "test.parquet"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Не найдена обученная LSTM: {ckpt_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Не найдена тестовая выборка: {parquet_path}")
    actual_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if actual_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна: используйте --device cpu")
    dev = torch.device(actual_device)
    print(f"  [LSTM узел {node}] инференс на {actual_device} (без переобучения)", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cols = list(ckpt["numeric_cols"])
    src_v = {int(k): int(v) for k, v in ckpt["src_vocab"].items()}
    dst_v = {int(k): int(v) for k, v in ckpt["dst_vocab"].items()}
    proto_v = {int(k): int(v) for k, v in ckpt["protocol_vocab"].items()}
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(cols)
    classes = ckpt["classes"]
    cfg = ckpt.get("config", {})
    model = module.LSTMClassifier(
        n_numeric=len(cols), src_cardinality=len(src_v) + 1,
        dst_cardinality=len(dst_v) + 1, protocol_cardinality=len(proto_v) + 1,
        hidden_size=cfg.get("hidden_size", module.HIDDEN_SIZE),
        num_layers=cfg.get("num_layers", module.NUM_LAYERS),
        dropout=cfg.get("dropout", module.DROPOUT),
        bidirectional=cfg.get("bidirectional", module.BIDIRECTIONAL),
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    seen = 0
    sequences, true_counts = [], []
    fields = list(dict.fromkeys(["window_id", "label"] + cols + module.FORCED_CATEGORICAL))
    max_len = cfg.get("max_flows_per_window", module.MAX_FLOWS_PER_WINDOW)
    collate = module.SequenceCollator(max_len=max_len)

    def score_batch():
        nonlocal seen
        if not sequences:
            return
        batch = [(*seq, 0) for seq in sequences]
        x_num, x_src, x_dst, x_proto, lengths, _ = collate(batch)
        with torch.inference_mode():
            logits = model(x_num.to(dev), x_src.to(dev), x_dst.to(dev),
                           x_proto.to(dev), lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
        for counts, pred in zip(true_counts, preds):
            cm[:, int(pred)] += counts
            seen += int(counts.sum())
        sequences.clear()
        true_counts.clear()

    n_rows = module._row_count(parquet_path)
    with tqdm(total=n_rows, desc=f"LSTM {node} test", unit="flow", unit_scale=True) as bar:
        for group in module._iter_groups(parquet_path, fields):
            y = group["label"].to_numpy(dtype=np.int64)
            if np.any((y < 0) | (y >= n_cls)):
                raise ValueError(f"Неверная метка в {parquet_path}")
            counts = np.bincount(y, minlength=n_cls)
            seq = module._group_to_sequence(group, cols, scaler, src_v, dst_v, proto_v,
                                            max_flows=max_len, compact=True)
            sequences.append(seq)
            true_counts.append(counts)
            if len(sequences) >= batch_size:
                score_batch()
            bar.update(len(group))
        score_batch()
    if seen != n_rows or cm.sum() != n_rows:
        raise AssertionError(f"LSTM подсчитала {seen} flow вместо {n_rows}")
    return cm, [str(c) for c in classes]


def _model_entries(project_root):
    """Находит отчёты в реальных папках, без фиксированных названий моделей."""
    models = project_root / "models"
    if models.is_dir():
        for path in sorted(models.rglob("metrics.json")):
            if path.is_file():
                name = path.parent.relative_to(models).as_posix()
                yield name, path, path.parent
    pipeline = project_root / "pipeline_results" / "metrics.json"
    if pipeline.is_file():
        yield "pipeline", pipeline, pipeline.parent


def _node_number(name, info):
    """Определить узел для необязательного LSTM-инференса."""
    node = info.get("node")
    if isinstance(node, int) and node in (1, 2):
        return node
    # Примеры: node1_lstm_2 (узел 2), node_lstm1 (узел 1).
    match = re.search(r"(?:_|\b)([12])$", name)
    if match and "lstm" in name.lower():
        return int(match.group(1))
    match = re.search(r"lstm[_-]?([12])$", name, flags=re.I)
    if match:
        return int(match.group(1))
    return None


def _get_counts(info):
    """Извлекает только НЕнормализованные количества потоков."""
    for key in ("confusion_matrix_counts", "confusion_matrix"):
        value = info.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("counts") or value.get("values")
        if value is None:
            continue
        cm = np.asarray(value)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            continue
        if np.all(np.isfinite(cm)) and np.all(cm >= 0) and np.all(cm == np.floor(cm)):
            # Нельзя выводить метрики по нормализованной матрице вместо количеств.
            # В более старых отчетах counts могут храниться просто как confusion_matrix.
            return cm.astype(np.int64)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."), help="Корень проекта")
    ap.add_argument("--out", type=Path, default=Path("gost_results"), help="Каталог общей таблицы")
    ap.add_argument("--skip-lstm-inference", action="store_true", help="Не восстанавливать CM из LSTM model.pt")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Устройство LSTM")
    ap.add_argument("--batch-size", type=int, default=64, help="Размер батча для LSTM инференса")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size должен быть положительным")
    root = args.root.resolve()
    output = args.out if args.out.is_absolute() else root / args.out
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    entries = list(_model_entries(root))
    if not entries:
        print(f"В {root / 'models'} не найдено ни одного metrics.json.")
        print("Укажи корень проекта с папкой models: python gost_evaluate.py --root ПУТЬ_К_ПРОЕКТУ")
        return
    print(f"Найдено файлов metrics.json: {len(entries)}", flush=True)
    for name, metrics_path, directory in entries:
        if not metrics_path.is_file():
            print(f"[ПРОПУСК] {name}: нет {metrics_path}")
            missing.append(name)
            continue
        try:
            info = json.loads(metrics_path.read_text(encoding="utf-8"))
            cm = _get_counts(info)
            # Только для LSTM с контрольной точкой возможен инференс без обучения.
            node = _node_number(name, info)
            is_lstm = ("lstm" in name.lower() or "lstm" in str(info.get("model", "")).lower())
            if cm is None and is_lstm and node is not None and (directory / "model.pt").is_file():
                if args.skip_lstm_inference:
                    print(f"[ПРОПУСК] {name}: в metrics.json нет confusion_matrix_counts")
                    missing.append(name)
                    continue
                cm, ckpt_classes = _lstm_counts(node, root, args.device, args.batch_size, directory)
                info["confusion_matrix_counts"] = cm.tolist()
                info.setdefault("classes", ckpt_classes)
                info["evaluation_unit"] = "flow"
                # Отдельно обновим файл отчёта модели, чтобы повторно не делать инференс.
                metrics_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            if cm is None:
                keys = ", ".join(sorted(info))
                print(f"[НЕТ МАТРИЦЫ] {name}: найден {metrics_path}")
                print(f"    Поля JSON: {keys}")
                print("    Нужна confusion_matrix_counts с числами потоков. Одного accuracy/F1 недостаточно ")
                print("    для вычисления Precision, Recall и Specificity по ГОСТ.")
                missing.append(name)
                continue
            raw_classes = info.get("classes") or []
            names = info.get("classes_ru") or raw_classes
            if not names or len(names) != np.asarray(cm).shape[0]:
                count = np.asarray(cm).shape[0]
                if count == 2 and (info.get("node") == 1 or name.lower().endswith("1")):
                    names = ["BENIGN", "ATTACK"]
                    raw_classes = ["BENIGN", "ATTACK"]
                else:
                    names = [f"Класс {i}" for i in range(count)]
                print(f"[ПОДПИСИ] {name}: использованы {names} (в JSON нет корректного classes/classes_ru)")
            result = evaluate_confusion_matrix(cm, names)
            result["model"] = info.get("model", name)
            result["name"] = name
            result["node"] = info.get("node", ("pipeline" if name == "pipeline" else name.rsplit("_", 1)[-1]))
            # Узел 1 — основной положительный класс ATTACK (вторая позиция).
            # Для узла 2 и финальной системы сравниваем macro по всем классам.
            attack = (result["per_class"][1] if len(raw_classes) == 2
                      and str(raw_classes[1]).upper() == "ATTACK" else None)
            if attack is not None:
                result["positive_class"] = attack["class"]
                result["positive_class_metrics"] = {
                    metric: attack[metric]
                    for metric in ("accuracy", "precision", "recall", "specificity", "f1")
                }
            save_reports(result, directory)
            row = {"model": name, "node": result["node"], "test_flows": result["test_flows"],
                   "correct_flows": result["correct_flows"], "accuracy": result["accuracy"],
                   "precision_macro": result["precision_macro"],
                   "recall_macro": result["recall_macro"],
                   "specificity_macro": result["specificity_macro"],
                   "f1_macro": result["f1_macro"], "f1_weighted": result["f1_weighted"],
                   "attack_precision": attack["precision"] if attack else None,
                   "attack_recall": attack["recall"] if attack else None,
                   "attack_specificity": attack["specificity"] if attack else None,
                   "attack_f1": attack["f1"] if attack else None}
            rows.append(row)
            shown = (attack if attack is not None else {
                "precision": row["precision_macro"], "recall": row["recall_macro"],
                "specificity": row["specificity_macro"], "f1": row["f1_macro"]
            })
            kind = "ATTACK" if attack is not None else "macro"
            print(f"[OK] {name:<19} flows={result['test_flows']:,} "
                  f"A={_as_display(row['accuracy'])} ({kind}) "
                  f"Pr={_as_display(shown['precision'])} Se={_as_display(shown['recall'])} "
                  f"Sp={_as_display(shown['specificity'])} "
                  f"F={_as_display(shown['f1'])}", flush=True)
        except (ValueError, KeyError, FileNotFoundError, RuntimeError, TypeError) as e:
            missing.append(name)
            print(f"[ОШИБКА] {name}: {e}", flush=True)
    if rows:
        with (output / "comparison_gost.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "comparison_gost.json").write_text(
            json.dumps({"standard": "ГОСТ Р 59898-2021, формулы 13–17",
                        "models": rows, "skipped": missing}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\nОбщее сравнение: {output / 'comparison_gost.csv'}")
        print("Для каждого алгоритма: models/<название>/gost_metrics.json и gost_metrics_by_class.csv")
    else:
        print("Файлы metrics.json найдены, но ни в одном нет пригодной матрицы с числами flow.")
    if missing:
        print("Не оценены (нет метрик/моделей или возникла ошибка): " + ", ".join(missing))


if __name__ == "__main__":
    main()
