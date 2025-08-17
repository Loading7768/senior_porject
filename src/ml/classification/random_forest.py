#!/usr/bin/env python3
"""
RandomForestClassifier with Feature Selection + 時間序切分

功能：
- 以『價格 npy』產生 y；支援 level/diff 模式
- 切分比率：60/20/20，時間序避免洩漏
- 支援 Feature Selection：
  none / kbest-chi2 / kbest-anova / kbest-mi / rfe-logreg / sfm-tree / sfm-l1
- 可手動排除 X 的 row（index 或區間）
- 可位移對齊 SHIFT_Y
- X/Y 不等長時可選對齊策略
- 輸出：Train/Val/Test 準確率、Test report、overfitting 圖、特徵重要度 JSON
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === utils for FS ===
from ml.utils.feature_selection import make_selector

# === 預設路徑 ===
FEATURES_DIR = Path("../data/keyword/machine_learning")
FEATURE_VECTOR_PATH = FEATURES_DIR / "feature_vector.npy"
FEATURE_NAMES_PATH = FEATURES_DIR / "feature_name.json"
DEFAULT_PRICE_NPY = Path("../data/coin_price/price_diff.npy")  # 預設 diff 模式

OUT_FIG = Path("../outputs/figures/ml/classification/rf_overfitting_check.png")
OUT_IMPORT_JSON = Path("../data/ml/classification/random_forest_feature_importances.json")
OUT_METRICS_TXT = Path("../data/ml/classification/random_forest_metrics.txt")

# ----------------- 輔助函式 -----------------
def build_labels(price_array: np.ndarray, mode: str) -> np.ndarray:
    p = np.asarray(price_array, dtype=float).reshape(-1)
    if mode == "level":
        diffs = p[1:] - p[:-1]
        return (diffs > 0).astype(int)
    elif mode == "diff":
        return (p > 0).astype(int)
    else:
        raise ValueError("price-mode 僅支援 'level' 或 'diff'")

def _parse_list_of_ints(csv_str: str) -> list[int]:
    if not csv_str.strip(): return []
    return [int(x) for x in csv_str.split(",") if x.strip()]

def _parse_slices(spec: str) -> list[tuple[int, int]]:
    if not spec.strip(): return []
    out = []
    for seg in spec.split(","):
        a, b = seg.split(":")
        out.append((int(a), int(b)))
    return out

def _apply_manual_exclusion_x(X: np.ndarray, indices, slices) -> np.ndarray:
    n = len(X); to_drop = set()
    for idx in indices:
        if 0 <= idx < n: to_drop.add(idx)
    for start, end in slices:
        to_drop.update(range(max(0,start), min(n,end)))
    if not to_drop: return X
    mask = np.ones(n, dtype=bool)
    mask[list(to_drop)] = False
    print(f"[Manual] Excluding {np.sum(~mask)} rows from X")
    return X[mask]

def _apply_shift_y(X, Y, shift):
    if shift == 0: return X, Y
    if shift > 0: return X[:-shift], Y[shift:]
    else: k = -shift; return X[k:], Y[:-k]

def _align_XY(X, Y, policy):
    if len(X) == len(Y): return X, Y
    m = min(len(X), len(Y))
    if policy=="keep_head": return X[:m], Y[:m]
    elif policy=="keep_tail": return X[-m:], Y[-m:]
    else: raise ValueError(f"lenX={len(X)}, lenY={len(Y)} 不符")

# ------------------------------- 主程式 -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-npy", type=Path, default=FEATURE_VECTOR_PATH)
    parser.add_argument("--feature-names", type=Path, default=FEATURE_NAMES_PATH)
    parser.add_argument("--price-npy", type=Path, default=DEFAULT_PRICE_NPY)
    parser.add_argument("--price-mode", choices=["level","diff"], default="diff")

    parser.add_argument("--exclude-x-indices", type=str, default="")
    parser.add_argument("--exclude-x-slices", type=str, default="")
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument("--align-policy", choices=["keep_tail","keep_head","error"], default="keep_tail")

    # RF 參數
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=32)
    parser.add_argument("--min-samples-split", type=int, default=64)
    parser.add_argument("--max-features", type=str, default="log2")
    parser.add_argument("--bootstrap", action="store_true", default=True)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--oob-score", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Feature selection 參數
    parser.add_argument("--fs", default="none",
                        choices=["none","kbest-chi2","kbest-anova","kbest-mi","rfe-logreg","sfm-tree","sfm-l1"])
    parser.add_argument("--k", type=int, default=600)
    parser.add_argument("--n-features", type=int, default=300)

    args = parser.parse_args()

    # === 讀取 X 與特徵名 ===
    X = np.load(args.features_npy)
    with open(args.feature_names,"r",encoding="utf-8-sig") as f:
        feature_names = json.load(f)

    # 手動排除
    indices = _parse_list_of_ints(args.exclude_x_indices)
    slices = _parse_slices(args.exclude_x_slices)
    if indices or slices: X = _apply_manual_exclusion_x(X, indices, slices)

    # 讀價格/變化量並產生 y
    price_arr = np.load(args.price_npy)
    y_all = build_labels(price_arr, args.price_mode)

    # SHIFT
    X, y_all = _apply_shift_y(X, y_all, args.shift_y)

    # level 模式下長度可能差1
    if args.price_mode=="level" and len(X)==len(y_all)+1: X=X[:-1]

    # 對齊
    X,Y = _align_XY(X, y_all, args.align_policy)

    # 時間序切分
    n=len(Y); i1=int(n*0.6); i2=int(n*0.8)
    X_train,y_train = X[:i1],Y[:i1]
    X_val,y_val     = X[i1:i2],Y[i1:i2]
    X_test,y_test   = X[i2:],Y[i2:]

    # === Feature Selection ===
    selected_feature_names = feature_names
    if args.fs!="none":
        fs = make_selector("clf", args.fs, k=args.k, n_features=args.n_features)
        fs.fit(X_train,y_train)
        X_train = fs.transform(X_train)
        X_val   = fs.transform(X_val)
        X_test  = fs.transform(X_test)
        if hasattr(fs,"get_support"):
            mask=fs.get_support()
            selected_feature_names=[n for n,keep in zip(feature_names,mask) if keep]
        else:
            selected_feature_names=[f"f{i}" for i in range(X_train.shape[1])]
        print(f"[FS] 方法={args.fs}, X_train shape={X_train.shape}")

    # === 訓練 RF ===
    rf=RandomForestClassifier(
        n_estimators=args.n_estimators,max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,min_samples_split=args.min_samples_split,
        max_features=args.max_features,bootstrap=args.bootstrap,max_samples=args.max_samples,
        oob_score=args.oob_score,random_state=args.seed,n_jobs=-1
    )
    rf.fit(X_train,y_train)

    # 評估
    train_acc=accuracy_score(y_train,rf.predict(X_train))
    val_acc=accuracy_score(y_val,rf.predict(X_val))
    test_pred=rf.predict(X_test)
    test_acc=accuracy_score(y_test,test_pred)
    print(f"Train acc: {train_acc:.4f}\nVal acc: {val_acc:.4f}\nTest acc: {test_acc:.4f}")
    report_str=classification_report(y_test,test_pred)
    print(report_str)

    # Overfitting 圖
    OUT_FIG.parent.mkdir(parents=True,exist_ok=True)
    plt.bar(["Train","Validation"],[train_acc,val_acc])
    plt.ylim(0,1); plt.ylabel("Accuracy"); plt.title("RF: Train vs Val")
    plt.tight_layout(); plt.savefig(OUT_FIG); plt.close()

    # 特徵重要度
    importances=rf.feature_importances_
    importance_series=pd.Series(importances,index=selected_feature_names).sort_values(ascending=False)
    OUT_IMPORT_JSON.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT_IMPORT_JSON,"w",encoding="utf-8") as f:
        json.dump({k:float(v) for k,v in importance_series.items()},f,ensure_ascii=False,indent=2)

    # 指標文字檔
    OUT_METRICS_TXT.parent.mkdir(parents=True,exist_ok=True)
    with open(OUT_METRICS_TXT,"w",encoding="utf-8") as f:
        f.write(f"X shape: {X.shape}, Y shape: {Y.shape}\n")
        f.write(f"Positive rate: {Y.mean():.3f}\n")
        f.write(f"Train acc: {train_acc:.3f}\nVal acc: {val_acc:.3f}\nTest acc: {test_acc:.3f}\n\n")
        f.write(report_str+"\n")
        f.write(f"圖: {OUT_FIG}\n重要度: {OUT_IMPORT_JSON}\n")

    print(f"圖已輸出：{OUT_FIG}\n特徵重要度 JSON：{OUT_IMPORT_JSON}\nMetrics：{OUT_METRICS_TXT}")

if __name__=="__main__":
    main()