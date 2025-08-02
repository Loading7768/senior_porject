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
    priceClose,date
    0.00231455,2018/08/22
    0.00238532,2018/08/23
    0.00242611,2018/08/24
    ...
'''

# === 修改為你的 CSV 檔與 JSON 資料夾路徑 ===
PRICE_CSV_PATH = f"../data/coin_price/{COIN_SHORT_NAME}_price.csv"
TWEETS_JSON_GLOB = f"../data/tweets/{COIN_SHORT_NAME}/*/*/*.json"
OUTPUT_CSV_PATH = "../data/coin_price/current_tweet_price_output.csv"

# === 讀取價格 CSV ===
price_df = pd.read_csv(PRICE_CSV_PATH)
price_df['date'] = pd.to_datetime(price_df['date'], format="%Y/%m/%d")
price_df.set_index('date', inplace=True)

# === 收集 tweet 有出現的日期 ===
tweet_dates = set()

json_files = glob(TWEETS_JSON_GLOB)
for json_path in json_files:
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
                price = row['priceClose'].values[0] if not row.empty else ""

                output_rows.append({
                    "date": d.strftime("%Y/%m/%d"),
                    "price": price,
                    "has_tweet": False
                })
    # 當前 tweet 日期
    price = price_df.loc[current_date]['priceClose'] if current_date in price_df.index else ""
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
