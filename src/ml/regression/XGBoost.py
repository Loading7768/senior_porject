import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


'''可修改參數'''
COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]

N = [0.004556786265841064, 179.99285067824763, 0.00032174181506763714]  # 分別要放大縮小的倍率

END_DATE = "2025-08-31"

INPUT_PATH = "../data/coin_price"

OUTPUT_PATH = "../data/ml/dataset/y_input"
'''可修改參數'''

os.makedirs(OUTPUT_PATH, exist_ok=True)

# df["Close"] *= 0.00032174181506763714  # TRUMP  R²=-0.396599
# df["Close"] *= 179.99285067824763      # PEPE   R²=0.835741
# df["Close"] *= 0.004556786265841064    # DOGE   R²=0.946255



best_r2 = -np.inf
best_n = None

# 或對數網格（範圍更廣，更常用）
# n_values = np.logspace(-5, 3, 1000)  # 從 1e-5 到 1e3，共 1000 個點

# for n in tqdm(n_values, desc="正在找尋最好的參數..."):
for coin_short_name, n in zip(COIN_SHORT_NAME, N):
    print(f"\n幣種：{coin_short_name}, 放大縮小倍率：{n}")

    # =============================
    # 1. 讀取資料
    # =============================
    df = pd.read_csv(f"{INPUT_PATH}/{coin_short_name}_price.csv")
    df = df.drop(columns=["market_cap", "total_volume"])

    # 只保留 Close
    df.rename(columns={"price": "Close"}, inplace=True)

    df_temp = df.copy()
    df_temp["Close"] *= n


    # print(df.head())

    # print(df["Close"].head(20))
    # print(df["Close"].diff().head(20))


    # print("Before dropna:", df.shape)
    df_temp = df_temp.dropna()
    # print("After dropna:", df.shape)

    # =============================
    # 2. 計算技術指標 (EMA, MACD, RSI)
    # =============================

    # EMA
    df_temp["EMA_12"] = df_temp["Close"].ewm(span=12, adjust=False).mean()
    df_temp["EMA_26"] = df_temp["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df_temp["MACD"] = df_temp["EMA_12"] - df_temp["EMA_26"]
    df_temp["Signal"] = df_temp["MACD"].ewm(span=9, adjust=False).mean()

    # RSI (SMA 版)
    def compute_RSI(series, period=14):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # # RSI (EMA 平滑版)
    # def compute_RSI(prices, period=14):
    #     delta = prices.diff()

    #     gain = delta.clip(lower=0)
    #     loss = -delta.clip(upper=0)

    #     # 使用 EMA 平滑
    #     avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    #     avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    #     rs = avg_gain / avg_loss
    #     rsi = 100 - (100 / (1 + rs))

    #     return rsi

    df_temp["RSI"] = compute_RSI(df_temp["Close"], 14)

    # =============================
    # 3. 準備訓練資料
    # =============================

    # 使用價格
    df_temp["Target"] = df_temp["Close"].shift(-1)

    # 使用價格變化率
    # df["Return"] = df["Close"].pct_change()
    # df["Target"] = df["Return"].shift(-1)


    # 移除 NaN
    df_temp = df_temp.dropna()

    features = ["EMA_12", "EMA_26", "MACD", "Signal", "RSI"]
    X = df_temp[features]
    y = df_temp["Target"]

    # =============================
    # 4. 切分訓練/測試集
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # print("X_train.var():", X_train.var())

    # =============================
    # 5. 訓練模型 (XGBoost)
    # =============================
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)

    # =============================
    # 6. 預測與評估
    # =============================
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print(f"\nn={n}, R²={r2:.6f}")

    # if r2 > best_r2:
    #     best_r2 = r2
    #     best_n = n

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R²:", r2)

    # =============================
    # 7. 準備 features 並存成 .npy
    # =============================
    features = ["EMA_12", "EMA_26", "MACD", "Signal", "RSI"]
    X_all = df_temp[features].to_numpy()  # shape (N, 5)

    # 只取日期部分 (YYYY-MM-DD)
    dates = pd.to_datetime(df_temp["snapped_at"]).dt.date.astype(str).to_numpy().reshape(-1, 1)  # shape (N, 1)

    # 🔹 新增：裁切到 2025-07-31
    cutoff = np.datetime64(END_DATE)
    mask = pd.to_datetime(dates.flatten()) <= cutoff

    X_all = X_all[mask]
    dates = dates[mask].reshape(-1, 1)

    print(X_all[:26])
    print(dates[:5])
    print(dates[-5:])
    print(f"{coin_short_name} 特徵 shape = {X_all.shape}")
    print(f"{coin_short_name} 日期 shape = {dates.shape}")

    # ===== 存成 npy / txt =====
    np.save(os.path.join(OUTPUT_PATH, coin_short_name, f"{coin_short_name}_XGBoost_features.npy"), X_all)
    np.savetxt(f"{OUTPUT_PATH}/{coin_short_name}/{coin_short_name}_XGBoost_dates.txt", dates, fmt="%s")

    # ===== 存成 CSV (合併日期+特徵) =====
    df_features = pd.DataFrame(
        np.hstack([dates, X_all]),
        columns=["Date"] + features
    )
    df_features.to_csv(f"{OUTPUT_PATH}/{coin_short_name}/{coin_short_name}_XGBoost_features.csv", index=False)

    print(f"✅ {coin_short_name} 已輸出 CSV：{OUTPUT_PATH}/{coin_short_name}/{coin_short_name}_XGBoost_features.csv")
    
    # =============================
    # 7. 畫圖比較
    # =============================
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test.index, y_test, label="True Price")
    # plt.plot(y_test.index, y_pred, label="Predicted Price")
    # plt.legend()
    # plt.xlabel("Date")
    # plt.ylabel("Price (USD)")
    # plt.title(f"{COIN_SHORT_NAME} Price Prediction with EMA, MACD, RSI")
    # plt.xticks(rotation=45)
    # plt.show()

print("\n✅ 全部幣種特徵已存成 .npy 檔")

# print(f"Best n={best_n}, Best R²={best_r2:.6f}")



