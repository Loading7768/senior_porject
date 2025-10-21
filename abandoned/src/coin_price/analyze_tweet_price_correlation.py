import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

# === 參數設定 ===
COIN = 'PEPE'
YEAR = '2025'
MONTH = '02'
FILENAME = f'{COIN}_{YEAR}_{MONTH}_tweet_price_summary.csv'

SUMMARY_PATH = f'../data/tweets/summary/{FILENAME}'
OUTPUT_FOLDER = '../outputs/figures/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === 讀取與整理資料 ===
df = pd.read_csv(SUMMARY_PATH)
df['normal_tweet_count'] = pd.to_numeric(df['normal_tweet_count'], errors='coerce')
df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
df.dropna(subset=['normal_tweet_count', 'close_price'], inplace=True)

# === 同步皮爾森相關分析 ===
correlation, p_value = pearsonr(df['normal_tweet_count'], df['close_price'])
print(f"【{FILENAME}】Correlation: {correlation:.4f}, p-value: {p_value:.4f}")

# === 畫同步散佈圖 ===
plt.figure(figsize=(8, 6))
plt.scatter(df['normal_tweet_count'], df['close_price'], alpha=0.7)
plt.title(f'{FILENAME}\nCorrelation: {correlation:.4f}, p-value: {p_value:.4f}')
plt.xlabel('Tweet Count')
plt.ylabel('Close Price')
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_FOLDER, f'{COIN}_{YEAR}{MONTH}_correlation_plot.png')
plt.savefig(plot_path)
print(f"同步相關圖表已儲存至：{plot_path}")
plt.close()

# === 滯後皮爾森相關分析（Lag -7 ~ +7） ===
max_lag = 7
lag_results = []

for lag in range(-max_lag, max_lag + 1):
    shifted_df = df.copy()
    shifted_df['shifted_price'] = shifted_df['close_price'].shift(-lag)
    shifted_df.dropna(subset=['normal_tweet_count', 'shifted_price'], inplace=True)

    if len(shifted_df) > 1:  # 至少需要兩個點才能計算相關
        corr, p = pearsonr(shifted_df['normal_tweet_count'], shifted_df['shifted_price'])
        lag_results.append((lag, corr, p))
    else:
        lag_results.append((lag, None, None))

# === 印出滯後分析結果 ===
print("\n=== 滯後相關係數分析（推文 vs 價格）===")
for lag, corr, p in lag_results:
    if corr is not None:
        print(f"Lag {lag:+}: Correlation = {corr:.4f}, p = {p:.4f}")
    else:
        print(f"Lag {lag:+}: 資料不足")

# === 畫滯後折線圖 ===
valid_lags = [r[0] for r in lag_results if r[1] is not None]
valid_corrs = [r[1] for r in lag_results if r[1] is not None]

plt.figure(figsize=(10, 5))
plt.plot(valid_lags, valid_corrs, marker='o')
plt.axhline(0, color='gray', linestyle='--')
plt.title(f'{FILENAME} - Tweet vs Price Lagged Correlation')
plt.xlabel('Lag (days)')
plt.ylabel('Pearson Correlation')
plt.grid(True)
plt.tight_layout()

lag_plot_path = os.path.join(OUTPUT_FOLDER, f'{COIN}_{YEAR}{MONTH}_lagged_correlation_plot.png')
plt.savefig(lag_plot_path)
print(f"滯後相關圖表已儲存至：{lag_plot_path}")
plt.close()
