#!/usr/bin/env python3
"""
- 以『價格 npy』產生 y；支援兩種模式：
  - level：把價格視為價位，y[t] = 1 若 price[t+1] > price[t] 否則 0
  - diff：把價格 npy 視為『日變化量』(例如 price_diff.npy)，y[t] = 1 若 diff[t] > 0 否則 0
- 切分比率：60%/20%/20% (train/val/test)，使用 train_test_split 並 stratify
- 使用 RandomForestClassifier，輸出：
  1) Train/Val/Test 準確率 + Test classification_report
  2) Overfitting 檢查圖 (Train vs Val bar chart)
  3) 特徵重要度 JSON（依特徵名排序）

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    balanced_accuracy_score,
)


# === 預設路徑（可用參數覆蓋） ===
FEATURES_DIR = Path("../data/keyword/machine_learning")
FEATURE_VECTOR_PATH = FEATURES_DIR / "feature_vector.npy"
FEATURE_NAMES_PATH = FEATURES_DIR / "feature_name.json"

DEFAULT_PRICE_NPY = Path("../data/coin_price/price_diff.npy")  # 預設當作 diff 模式

OUT_FIG = Path("../outputs/figures/ml/classification/rf_overfitting_check.png")
OUT_IMPORT_JSON = Path("../data/ml/classification/random_forest_feature_importances.json")
OUT_METRICS_TXT = Path("../data/ml/classification/random_forest_metrics.txt")


def build_labels(price_array: np.ndarray, mode: str) -> np.ndarray:
    """由價格（或日變化量）序列產生二元標籤。
    mode='level'：輸入為價位，輸出長度 N-1；y[t]=1 若 price[t+1]>price[t]
    mode='diff' ：輸入為當日變化量，輸出長度 N   ；y[t]=1 若 diff[t]>0
    """
    p = np.asarray(price_array, dtype=float).reshape(-1)
    if mode == "level":
        if p.size < 2:
            raise ValueError("價格 npy 長度需 ≥ 2 才能計算明日漲跌 (level 模式)")
        diffs = p[1:] - p[:-1]
        return (diffs > 0).astype(int)
    elif mode == "diff":
        return (p > 0).astype(int)
    else:
        raise ValueError("price-mode 僅支援 'level' 或 'diff'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-npy", type=Path, default=FEATURE_VECTOR_PATH, help="feature_vector.npy 路徑")
    parser.add_argument("--feature-names", type=Path, default=FEATURE_NAMES_PATH, help="feature_name.json 路徑")
    parser.add_argument("--price-npy", type=Path, default=DEFAULT_PRICE_NPY, help="價格/日變化量 npy 路徑")
    parser.add_argument("--price-mode", choices=["level", "diff"], default="diff", help="price npy 的語意：價位(level) 或 變化量(diff)")

    # RF 主要超參數
    parser.add_argument("--n-estimators", type=int, default=500, help="樹的數量")
    parser.add_argument("--max-depth", type=int, default=None, help="最大深度，預設 None")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="葉節點最小樣本數")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced", help="類別權重")

    parser.add_argument("--seed", type=int, default=42, help="亂數種子")

    args = parser.parse_args()

    # === 讀取 X 與特徵名 ===
    X = np.load(args.features_npy)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(f"{args.features_npy} 不是 2D numpy 陣列，實際：{type(X)}, ndim={getattr(X, 'ndim', None)}")

    with open(args.feature_names, "r", encoding="utf-8-sig") as f:
        feature_names = json.load(f)
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"feature_name.json 長度({len(feature_names)})與 X 特徵數({X.shape[1]})不符")

    # === 讀取價格/變化量 npy 並產生 y ===
    price_arr = np.load(args.price_npy)
    y_all = build_labels(price_arr, mode=args.price_mode)

    # === 對齊 X 與 y ===
    # level 模式：y 長度 = N-1；diff 模式：y 長度 = N
    m = min(len(X), len(y_all))
    X = X[:m]
    Y = y_all[:m]

    # 時間序 60/20/20 切分（避免資訊洩漏）
    n = len(Y)
    i1 = int(n * 0.6)
    i2 = int(n * 0.8)
    X_train, y_train = X[:i1], Y[:i1]
    X_val,   y_val   = X[i1:i2], Y[i1:i2]
    X_test,  y_test  = X[i2:],   Y[i2:]


    # === 建立並訓練 RandomForest ===
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=4,            # ← 原本 None，改小幅度限制樹深度
        min_samples_leaf=32,    # ← 葉子至少 32 筆，避免記憶訓練集
        min_samples_split=64,   # ← 節點至少 64 筆才分裂，進一步抑制過擬合
        max_features='log2',    # ← 每次分裂只看更少特徵，降方差
        bootstrap=True,         # ← 預設就是 True，寫明沒壞處
        max_samples=0.7,        # ← 每棵樹只抽 70% 資料，增加多樣性
        class_weight=None,      # ← 你的類別比例 7:5 不嚴重，先關掉
        oob_score=True,         # ← 看袋外分數，輔助檢查泛化
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)


    # === 評估 ===
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_val, rf.predict(X_val))
    test_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Train 準確率: {train_acc:.4f}")
    print(f"Validation 準確率: {val_acc:.4f}")
    print(f"Test 準確率: {test_acc:.4f}")

    print("\n分類報告 (Test set):")
    report_str = classification_report(y_test, test_pred)
    print(report_str)

    # === 儲存 Overfitting 圖 ===
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(["Train", "Validation"], [train_acc, val_acc])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("RandomForest: Train vs Validation Accuracy")
    plt.tight_layout()
    plt.savefig(OUT_FIG)
    plt.close()

    # === 特徵重要度輸出 ===
    importances = getattr(rf, "feature_importances_", None)
    if importances is None:
        raise RuntimeError("此 RandomForest 模型沒有 feature_importances_ 屬性")

    importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    OUT_IMPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_IMPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in importance_series.items()}, f, ensure_ascii=False, indent=2)

    # === 指標輸出到文字檔 ===
    OUT_METRICS_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(f"X shape: {X.shape}, Y shape: {Y.shape}\n")
        if len(Y) > 0:
            pos_rate = Y.mean()
            f.write(f"Positive rate (Y==1): {pos_rate:.3f}\n")
        f.write("\n[Accuracy]\n")
        f.write(f"Train: {train_acc:.3f}\nVal: {val_acc:.3f}\nTest: {test_acc:.3f}\n\n")
        f.write("[Classification Report - Test]\n")
        f.write(report_str + "\n")
        f.write(f"Overfitting 圖: {OUT_FIG}\n")
        f.write(f"特徵重要度 JSON: {OUT_IMPORT_JSON}\n")

    print(f"圖已輸出：{OUT_FIG}")
    print(f"特徵重要度 JSON：{OUT_IMPORT_JSON}")
    print(f"已輸出評估結果：{OUT_METRICS_TXT}")


if __name__ == "__main__":
    main()
