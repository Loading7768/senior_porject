from collections import defaultdict
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

'''
要先把價錢和日期的 csv 檔放在 ../data/coin_price 中
檔名存為 {COIN_SHORT_NAME}_price.csv

DOGE_price.csv：
    snapped_at,price,market_cap,total_volume
    2014-01-28 00:00:00 UTC,0.00134193,50833265.0,2342040.0
    2014-01-29 00:00:00 UTC,0.00138856,53584322.0,1655470.0
    2014-01-30 00:00:00 UTC,0.00147485,58009736.0,2315200.0
    ...
'''

# === 修改為你的 CSV 檔與 JSON 資料夾路徑 ===
PRICE_CSV_PATH = f"../data/coin_price/{COIN_SHORT_NAME}_price.csv"
NORMAL_TWEETS_JSON_GLOB = f"../data/filtered_tweets/normal_tweets/*/*/*.json"  # 是針對 normal_tweet 做運算
OUTPUT_CSV_PATH = f"../data/coin_price/{COIN_SHORT_NAME}_current_tweet_price_output.csv"

# === 讀取價格 CSV ===
price_df = pd.read_csv(PRICE_CSV_PATH)
price_df['snapped_at'] = pd.to_datetime(price_df['snapped_at'], format="%Y-%m-%d %H:%M:%S %Z")
price_df.set_index('snapped_at', inplace=True)
price_df.index = price_df.index.tz_localize(None)  # 移除時區 只保留日期部分

# === 儲存推文資訊 若當天沒有推文則不會加進去 set 中 ===
tweet_dates = set()  # 收集 tweet 有出現的日期

tweet_count = defaultdict(int)  # 儲存每天的推文數量


json_files = glob(NORMAL_TWEETS_JSON_GLOB)
for json_path in tqdm(json_files, desc="正在找尋日期"):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    tweets = data[JSON_DICT_NAME]
    if not tweets:
        continue

    try:
        # 取得日期
        date_str = datetime.strptime(
            tweets[0]['created_at'], "%a %b %d %H:%M:%S %z %Y"
        ).strftime("%Y/%m/%d")
        date_dt = pd.to_datetime(date_str)
        tweet_dates.add(date_dt)

        # 取得當天推文數量
        tweet_count[date_dt] = len(tweets)

    except Exception as e:
        print(f"[錯誤] {json_path}: {e}")

# === 依照 tweet 日期排序，決定整個時間範圍 ===
if not tweet_dates:
    print("沒有抓到任何推文日期")
    exit()

tweet_dates = sorted(tweet_dates)  # 因為抓進來的檔案順序可能會是亂的

# ----------- 將 tweet_count 輸出成 json 檔 -------------
# 將 datetime 轉成字串，defaultdict -> dict
tweet_count_dict = {date.strftime("%Y/%m/%d"): count for date, count in tweet_count.items()}

# 儲存成 JSON
output_tweet_count_path = f"../data/coin_price/{COIN_SHORT_NAME}_current_tweet_count.json"
with open(output_tweet_count_path, "w", encoding="utf-8") as f:
    json.dump(tweet_count_dict, f, ensure_ascii=False, indent=4)

print(f"✅ 已儲存 {COIN_SHORT_NAME}_tweet_count 到 {output_tweet_count_path}")

total_tweets = sum(tweet_count.values())
print(f"\n全部 normal_tweet 的推文數量: {total_tweets}\n")

# === 建立最終結果表 ===
output_rows = []

prev_date = None
for current_date in tqdm(tweet_dates, desc="正在儲存價錢"):
    if prev_date:

        # 若有缺少的日期 且 相鄰兩天間少於 31 天
        gap = (current_date - prev_date).days
        if 1 < gap < 31:
            for d in pd.date_range(prev_date + timedelta(days=1), current_date - timedelta(days=1)):
                
                row = price_df.loc[price_df.index == d]
                price = row['price'].values[0] if not row.empty else ""

                output_rows.append({
                    "date": d.strftime("%Y/%m/%d"),
                    "price": price,
                    "tweet_count": 0,
                    "has_tweet": False
                })
    # 當前 tweet 日期
    price = price_df.loc[current_date]['price'] if current_date in price_df.index else ""
    output_rows.append({
        "date": current_date.strftime("%Y/%m/%d"),
        "price": price,
        "tweet_count": tweet_count[current_date],
        "has_tweet": True
    })
    prev_date = current_date

# 將 output_rows 轉成 DataFrame
df_output = pd.DataFrame(output_rows)

# 將 date 字串轉成 datetime（方便計算隔天日期）
df_output['date_dt'] = pd.to_datetime(df_output['date'], format='%Y/%m/%d')

# 定義一個函數計算 price 差
def calc_price_diff(row):
    today = row['date_dt']
    tomorrow = today + pd.Timedelta(days=1)
    try:
        price_today = row['price']
        price_tomorrow = price_df.loc[tomorrow]['price']
        if price_today == "" or pd.isna(price_today):
            return ""
        return price_tomorrow - price_today
    except KeyError:
        return ""  # 隔天沒價格就空字串

# 計算差價欄位
df_output['price_diff'] = df_output.apply(calc_price_diff, axis=1)

# 刪除輔助欄位
df_output.drop(columns=['date_dt'], inplace=True)

# 儲存原本的 CSV
df_output.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 已儲存到 {OUTPUT_CSV_PATH}")

# 將 price_diff 欄位轉為 float，NaN 就會顯現出來
df_output['price_diff_float'] = pd.to_numeric(df_output['price_diff'], errors='coerce')

# 找出 NaN 的列
nan_rows = df_output[df_output['price_diff_float'].isna()]

# 顯示是哪幾天
print("\n以下日期 price_diff 無法計算（可能缺少當天或隔天價格）:")
print(nan_rows[['date', 'price', 'tweet_count', 'has_tweet']])


# ----------------------- 儲存 price_diff.npy --------------------------
# 先把空字串轉成 NaN，方便處理（這一步會將非數值都轉成 NaN）
df_output['price_diff'] = pd.to_numeric(df_output['price_diff'], errors='coerce')

# 過濾出 has_tweet == True 的資料，且 price_diff 不是 NaN
filtered_df = df_output[(df_output['has_tweet'] == True) & (df_output['price_diff'].notna())]

# 依 tweet_count 重複 price_diff
expanded_price_diffs = []
for _, row in filtered_df.iterrows():  # 逐行遍歷 filtered_df
    expanded_price_diffs.extend([row['price_diff']] * row['tweet_count'])  # extend() 方法會把這個 list 的所有元素加到 expanded_price_diffs 裡

price_diff_array = np.array(expanded_price_diffs, dtype=float)  # 轉成 numpy 陣列
np.save(f"../data/coin_price/{COIN_SHORT_NAME}_price_diff.npy", price_diff_array)  # 存成 .npy 檔

# 顯示預覽
print(f"\n✅ 已儲存 {COIN_SHORT_NAME}_price_diff.npy（共 {len(price_diff_array)} 筆）：\n{price_diff_array}")

