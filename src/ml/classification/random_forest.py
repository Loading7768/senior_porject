#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === 固定檔案路徑（可改） ===
FEATURES_PATH = Path("../data/keyword/machine_learning/feature_vector.npy")
PRICES_PATH = Path("../data/coin_price/pepe_price.csv")
FEATURE_NAMES_PATH = Path("../data/keyword/machine_learning/feature_name.json")
OUT_JSON = Path("../data/keyword/machine_learning/feature_importance.json")
OUT_METRICS = Path("../data/keyword/machine_learning/metrics.txt")


# -------------------------------
# I/O 與資料讀取工具
# -------------------------------
def _smart_read_csv(path: Path) -> pd.DataFrame:
    """嘗試自動辨識分隔符（逗號/Tab），不行就用預設。"""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path)


def load_prices(csv_path: Path) -> pd.DataFrame:
    """讀取價格 CSV，支援逗號或 tab 分隔，回傳欄位 ['Date','priceOpen']"""
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(csv_path, sep="\t")
        except Exception:
            df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]
    date_col = None
    price_col = None
    for c in df.columns:
        lc = c.lower()
        if "date" in lc and date_col is None:
            date_col = c
        if ("priceopen" in lc or "open" in lc or "price" in lc) and price_col is None:
            price_col = c

    if date_col is None or price_col is None:
        raise ValueError(f"找不到日期/價格欄位：{df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date", price_col: "priceOpen"})
    df["Date"] = df["Date"].dt.normalize()  # 僅保留日期
    return df[["Date", "priceOpen"]]


def load_feature_dates(path: Path) -> pd.Series:
    """
    讀取每一列特徵對應的日期（長度要和 X 的列數相同）。
    - 若有表頭，會找含 'date' 的欄位。
    - 若無表頭，視為單欄日期。
    回傳：pd.Series(datetime64[ns])（已 normalize 至日）
    """
    df = _smart_read_csv(path)
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    s = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    if s.isna().any():
        raise ValueError("feature dates 內含無法解析的日期（NaT），請檢查檔案內容。")
    return s


def load_tweet_status(path: Path) -> pd.DataFrame:
    """
    讀推文狀態 CSV（需要 'date' 與 'has_tweet' 欄位）。
    - date 會被轉成 datetime（只看到日）
    - has_tweet 會被轉成布林（支援 1/0/true/false）
    回傳：DataFrame[['date','has_tweet']]
    """
    df = _smart_read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns:
        for c in list(df.columns):
            if "date" in c:
                df = df.rename(columns={c: "date"})
                break
    if "has_tweet" not in df.columns:
        for c in list(df.columns):
            if "tweet" in c and "has" in c:
                df = df.rename(columns={c: "has_tweet"})
                break

    if "date" not in df.columns or "has_tweet" not in df.columns:
        raise ValueError("tweet_status CSV 需要 'date' 與 'has_tweet' 欄位")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["has_tweet"] = df["has_tweet"].astype(str).str.strip().str.lower().map(
        {"1": True, "true": True, "0": False, "false": False}
    ).fillna(False)

    df = df.dropna(subset=["date"]).loc[:, ["date", "has_tweet"]]
    return df


# -------------------------------
# 建立 y 與模型管線
# -------------------------------
def make_labels_from_prices(prices: np.ndarray) -> np.ndarray:
    """y[t] = 1 若明天價格比今天高，否則 0（長度 = len(prices)-1）"""
    diffs = prices[1:] - prices[:-1]
    return (diffs > 0).astype(int)


def build_pipeline(model: str):
    if model == "rf":
        # 樹模型不需要標準化
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        return Pipeline([("clf", clf)])
    else:
        # 預設：Logistic Regression + StandardScaler
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])


def extract_importance(pipe, feature_names, model: str) -> dict:
    if model == "rf":
        importances = pipe.named_steps["clf"].feature_importances_
    else:
        coefs = pipe.named_steps["clf"].coef_.ravel()
        abs_coefs = np.abs(coefs)
        importances = abs_coefs / abs_coefs.sum() if abs_coefs.sum() > 0 else abs_coefs
    return {k: float(v) for k, v in sorted(
        zip(feature_names, importances), key=lambda kv: kv[1], reverse=True
    )}


# -------------------------------
# 主流程
# -------------------------------
def main(args):
    # 讀取特徵矩陣 X
    X = np.load(FEATURES_PATH, allow_pickle=True)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(f"{FEATURES_PATH} 不是 2D numpy 陣列，實際：{type(X)}, ndim={getattr(X, 'ndim', None)}")

    # 讀取價格
    price_df = load_prices(PRICES_PATH)  # ['Date','priceOpen']

    # 預設將用完整價格序列（會在下面再和 X 對齊）
    prices = price_df["priceOpen"].to_numpy()

    # === 只使用有推文的日期（可選） ===
    filtered_days = None
    before_rows = len(X)

    if args.use_has_tweet_only:
        if args.feature_dates is None or args.tweet_status is None:
            raise ValueError("--use-has-tweet-only 需要同時提供 --feature-dates 與 --tweet-status")
        # 1) 讀 X 的日期（需與 X 列數相同）
        feature_dates = load_feature_dates(args.feature_dates)
        if len(feature_dates) != len(X):
            raise ValueError(f"feature_dates 長度({len(feature_dates)})需與 X 列數({len(X)})一致")
        feature_dates = feature_dates.dt.normalize()

        # 2) 讀 tweet 狀態，挑 has_tweet==True 的日期們
        ts = load_tweet_status(args.tweet_status)
        processed_dates = set(ts.loc[ts["has_tweet"], "date"].tolist())

        # 3) 用 has_tweet 過濾 X
        mask_has_tweet = feature_dates.isin(processed_dates)
        X = X[mask_has_tweet]
        feature_dates = feature_dates[mask_has_tweet].reset_index(drop=True)

        # 4) 與價格內連接，只保留雙方都有的日期，並依日期排序
        idx_after_mask = np.arange(len(feature_dates))
        fd = pd.DataFrame({"Date": feature_dates, "idx": idx_after_mask})
        join_df = (
            fd.merge(price_df, on="Date", how="inner")
              .sort_values("Date")
              .reset_index(drop=True)
        )
        if join_df.empty:
            raise ValueError("合併後沒有重疊日期（has_tweet=True 且有價格）。")

        # 5) 依 join 後日期順序重排 X 與 prices
        X = X[join_df["idx"].to_numpy()]
        prices = join_df["priceOpen"].to_numpy()
        filtered_days = len(X)
    else:
        # 未提供日期資訊時，假設 X 與價格皆為時間序且尾端對齊
        m = min(len(X), len(prices))
        X = X[-m:]
        prices = prices[-m:]

    # === 產生 y（明日漲跌），並與 X 對齊 ===
    y = make_labels_from_prices(prices)  # 長度 = len(prices)-1
    if len(X) != len(prices):
        # 在「只用有推文」情境已經對齊；一般情況做保守對齊
        m = min(len(X), len(prices))
        X = X[:m]
        prices = prices[:m]
        y = make_labels_from_prices(prices)

    # 丟掉最後一天的 X（因沒有「明天」可比較）
    if len(X) != len(y) + 1:
        # 容錯處理，保守裁切到最一致的長度
        k = min(len(X) - 1, len(y))
        X = X[:k + 1]
        y = y[:k]
    X = X[:-1]

    # 讀特徵名稱
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"feature_name.json 長度({len(feature_names)})與 X 特徵數({X.shape[1]})不符")

    # === 時間序切分：不打亂 ===
    n = len(y)
    test_size = int(round(0.2 * n))
    temp_size = n - test_size

    X_temp, X_test = X[:temp_size], X[temp_size:]
    y_temp, y_test = y[:temp_size], y[temp_size:]

    val_size = int(round(0.25 * temp_size))  # 25% of temp -> 20% of total
    train_size = temp_size - val_size

    X_train, X_val = X_temp[:train_size], X_temp[train_size:]
    y_train, y_val = y_temp[:train_size], y_temp[train_size:]

    assert len(X_train) + len(X_val) + len(X_test) == n

    # === 建立模型並訓練 ===
    pipe = build_pipeline(args.model)

    # 先在 train 上 fit，觀察 validation
    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred, digits=3)

    # 特徵重要度（用 val 訓練的模型）
    importance_map = extract_importance(pipe, feature_names, args.model)

    # 最終：用 train+val 重訓，報告 test
    X_trval = np.vstack([X_train, X_val])
    y_trval = np.hstack([y_train, y_val])

    pipe.fit(X_trval, y_trval)
    y_test_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred, digits=3)

    # === 輸出 ===
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(importance_map, f, ensure_ascii=False, indent=2)

    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write(f"X shape: {X.shape}, y shape: {y.shape}\n")
        if filtered_days is not None:
            f.write(f"Use has_tweet only: before={before_rows}, after={filtered_days}\n")
        f.write(f"Split -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}\n")
        if len(y) > 0:
            pos_rate = y.mean()
            f.write(f"Positive rate (y==1): {pos_rate:.3f}\n")
        f.write("\n[Validation]\n")
        f.write(f"Accuracy: {val_acc:.3f}\n")
        f.write(val_report + "\n")
        f.write("[Test]\n")
        f.write(f"Accuracy: {test_acc:.3f}\n")
        f.write(test_report)

    print(f"已儲存特徵重要度到 {OUT_JSON}")
    print(f"已儲存評估結果到 {OUT_METRICS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg",
                        help="選擇模型：logreg（預設）或 rf（RandomForest）")
    parser.add_argument("--feature-dates", type=Path, default=None,
                        help="對應 X 每一列的日期清單（CSV/TSV，可無表頭；若有表頭找含 'date' 的欄位）")
    parser.add_argument("--tweet-status", type=Path, default=None,
                        help="推文狀態 CSV，需含 'date' 與 'has_tweet' 欄位（1/0 或 true/false）")
    parser.add_argument("--use-has-tweet-only", action="store_true",
                        help="只使用 has_tweet==True 的日期來對齊 X 與價格")
    args = parser.parse_args()
    main(args)
