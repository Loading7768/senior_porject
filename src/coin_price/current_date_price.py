import os
import json
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from tqdm import tqdm

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
NORMAL_TWEETS_JSON_GLOB = f"../data/filtered_tweets/normal_tweets/*/*/*.json"
OUTPUT_CSV_PATH = "../data/coin_price/current_tweet_price_output.csv"

# === 讀取價格 CSV ===
price_df = pd.read_csv(PRICE_CSV_PATH)
price_df['snapped_at'] = pd.to_datetime(price_df['snapped_at'], format="%Y-%m-%d %H:%M:%S %Z")
price_df.set_index('snapped_at', inplace=True)
price_df.index = price_df.index.tz_localize(None)  # 移除時區 只保留日期部分

# === 收集 tweet 有出現的日期 ===
tweet_dates = set()

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
    except Exception as e:
        print(f"[錯誤] {json_path}: {e}")

# === 依照 tweet 日期排序，決定整個時間範圍 ===
if not tweet_dates:
    print("沒有抓到任何推文日期")
    exit()

min_date = min(tweet_dates)
max_date = max(tweet_dates)
tweet_dates = sorted(tweet_dates)

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
                    "has_tweet": False
                })
    # 當前 tweet 日期
    price = price_df.loc[current_date]['price'] if current_date in price_df.index else ""
    output_rows.append({
        "date": current_date.strftime("%Y/%m/%d"),
        "price": price,
        "has_tweet": True
    })
    prev_date = current_date

# === 儲存為 CSV ===
df_output = pd.DataFrame(output_rows)
df_output.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 已儲存到 {OUTPUT_CSV_PATH}")
