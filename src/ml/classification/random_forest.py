#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import scipy.sparse as sp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report

# === 檔案路徑 ===
DATASET_DIR = Path("../data/ml/dataset")
X_TRAIN_PATH = DATASET_DIR / "X_train_filtered.npz"
Y_TRAIN_PATH = DATASET_DIR / "Y_train_filtered.npz"
X_TEST_PATH = DATASET_DIR / "X_test.npz"
Y_TEST_PATH = DATASET_DIR / "Y_test.npz"
# === 載入推文 ID 對應表 ===
with open(f"{DATASET_DIR}/ids_train.pkl", "rb") as f:   # rb = read binary
    ids_train = pickle.load(f)
with open(f"{DATASET_DIR}/ids_test.pkl", "rb") as f:   # rb = read binary
    ids_test = pickle.load(f)
FEATURE_NAMES_PATH = Path("../data/ml/dataset/keyword/filtered_keywords.json")

OUT_FIG = Path("../outputs/figures/ml/classification/random_forest_overfitting_check.png")
OUT_IMPORTANCE_JSON = Path("../data/ml/classification/random_forest_keyword_importances.json")
OUT_METRICS_TXT = Path("../data/ml/classification/random_forest_metrics.txt")
OUT_DAILY_VOTE_PRED = Path("../outputs/ml/daily_vote_prediction.json")


def _predict_in_batches(model, X, batch_size=50000, desc="Predict"):
    n = X.shape[0]
    out = []
    for start in tqdm(range(0, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        out.append(model.predict(X[start:end]))
    return np.concatenate(out, axis=0)


def group_predictions_by_day(preds: list[int], dates: list[str]) -> dict:
    daily_votes = defaultdict(list)
    for p, d in zip(preds, dates):
        daily_votes[d].append(p)
    summary = {}
    for date, votes in daily_votes.items():
        cnt = Counter(votes)
        total = sum(cnt.values())
        up = cnt.get(1, 0) / total
        down = cnt.get(0, 0) / total
        summary[date] = {
            "total_votes": total,
            "up_votes": cnt.get(1, 0),
            "down_votes": cnt.get(0, 0),
            "up_ratio": round(up, 4),
            "down_ratio": round(down, 4),
            "majority": "up" if up > down else "down"
        }
    return summary


def _smart_load_matrix(path: Path):
    """Robustly load dense or sparse matrices from .npy/.npz.
    - .npy -> ndarray
    - .npz saved via scipy.sparse.save_npz -> reconstruct csr_matrix
    - .npz generic -> try common keys; otherwise first key
    Returns ndarray or scipy.sparse matrix.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=False)
        try:
            keys = list(z.files)
            print(f"Loaded {path.name} with keys: {keys}")
            # Detect scipy.sparse.save_npz format
            if all(k in keys for k in ["data", "indices", "indptr", "shape"]):
                data = z["data"]
                indices = z["indices"]
                indptr = z["indptr"]
                shape = tuple(z["shape"])  # shape stored as array
                mat = sp.csr_matrix((data, indices, indptr), shape=shape)
                return mat
            # Otherwise assume generic np.savez
            for k in ["X", "data", "array", "arr_0"]:
                if k in z:
                    return z[k]
            return z[keys[0]]
        finally:
            z.close()
    else:
        raise ValueError(f"Unsupported file extension for {path}")
    """Robustly load .npz/.npy.
    - .npy -> returns ndarray
    - .npz -> if single key, return that array; otherwise try common keys then the first one.
    Prints keys to help debugging.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        return arr
    elif path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=False)
        try:
            keys = list(z.files)
            if not keys:
                raise ValueError(f"{path} is empty .npz")
            print(f"Loaded {path.name} with keys: {keys}")
            # preferred keys
            for k in ["X", "data", "array", "arr_0"]:
                if k in z:
                    return z[k]
            # fall back to first key
            return z[keys[0]]
        finally:
            z.close()
    else:
        raise ValueError(f"Unsupported file extension for {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-mode", choices=["auto", "sign", "threshold"], default="auto")
    parser.add_argument("--label-threshold", type=float, default=0.0)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=50000)
    args = parser.parse_args()

    # === 讀取資料 ===
    X_train = _smart_load_matrix(X_TRAIN_PATH)
    data1 = np.load(Y_TRAIN_PATH)
    y_train = data1["Y"][:,4]
    X_test = _smart_load_matrix(X_TEST_PATH)
    data2 = np.load(Y_TEST_PATH)
    y_test = data2["Y"][:,4]
    dates_test = [date for _, date, _ in ids_test]

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8-sig") as f:
        feature_names = json.load(f)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
        verbose=args.verbose,
    )
    # === 將連續標籤轉成二元 {0,1} ===
    def _binarize(vec, mode="auto", thr=0.0):
        vec = np.asarray(vec).reshape(-1)
        if mode == "sign":
            return (vec > 0).astype(int)
        if mode == "threshold":
            return (vec > thr).astype(int)
        # auto
        uniq = np.unique(vec)
        if set(uniq).issubset({0,1}):
            return vec.astype(int)
        if set(uniq).issubset({-1,0,1}):
            return (vec > 0).astype(int)
        return (vec > 0).astype(int)

    y_train = _binarize(y_train, args.label_mode, args.label_threshold)
    y_test = _binarize(y_test, args.label_mode, args.label_threshold)
    print(f"Classes in y_train: {np.unique(y_train)}")

    model.fit(X_train, y_train)

    # === 預測 ===
    train_preds = _predict_in_batches(model, X_train, batch_size=args.batch_size, desc="Predict train")
    test_preds = _predict_in_batches(model, X_test, batch_size=args.batch_size, desc="Predict test")

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Train 準確率: {train_acc:.4f}")
    print(f"Test 準確率: {test_acc:.4f}")

    print("\n分類報告 (Test set):")
    report_str = classification_report(y_test, test_preds)
    print(report_str)

    # === 每天推文的票數與比率統計（Train/Test） ===
    dates_train = [date for _, date, _ in ids_train]
    dates_test = [date for _, date, _ in ids_test]

    train_daily_result = group_predictions_by_day(train_preds, dates_train)
    test_daily_result = group_predictions_by_day(test_preds, dates_test)

    # === 輸出 JSON 統計檔 ===
    output_dir = Path("../outputs/ml")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "random_forest_train_daily_results.json", "w", encoding="utf-8") as f:
        json.dump(train_daily_result, f, ensure_ascii=False, indent=2)

    with open(output_dir / "random_forest_test_daily_results.json", "w", encoding="utf-8") as f:
        json.dump(test_daily_result, f, ensure_ascii=False, indent=2)

    print("每天的預測統計已儲存至：")
    print(f"- train: {output_dir / 'random_forest_train_daily_results.json'}")
    print(f"- test : {output_dir / 'random_forest_test_daily_results.json'}")


    # === 畫 Overfitting 圖 ===
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Test"], [train_acc, test_acc])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Random Forest: Train vs Test Accuracy")
    plt.tight_layout()
    plt.savefig(OUT_FIG)
    plt.close()

    # === 特徵重要性輸出 ===
    importances = model.feature_importances_
    importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    OUT_IMPORTANCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_IMPORTANCE_JSON, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in importance_series.items()}, f, ensure_ascii=False, indent=4)

    # === 評估報告輸出 ===
    OUT_METRICS_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(f"X_train shape: {X_train.shape}, Y_train shape: {y_train.shape}\n")
        f.write(f"X_test shape: {X_test.shape}, Y_test shape: {y_test.shape}\n")
        pos_rate = float(np.mean(y_test))
        f.write(f"Positive rate (Y==1): {pos_rate:.3f}\n")
        f.write("\n[Accuracy]\n")
        f.write(f"Train: {train_acc:.3f}\nTest: {test_acc:.3f}\n\n")
        f.write("[Classification Report - Test]\n")
        f.write(report_str + "\n")
        f.write(f"Overfitting 圖: {OUT_FIG}\n")
        f.write(f"關鍵字 importantce JSON: {OUT_IMPORTANCE_JSON}\n")

if __name__ == "__main__":
    main()