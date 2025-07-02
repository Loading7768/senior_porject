import json
import os
import pandas as pd

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import config

'''可修改參數'''
COIN_YEAR = "2025"

COIN_MONTH = "03"

COIN_SHORT_NAME = config.COIN_SHORT_NAME

JSON_DICT_NAME = config.JSON_DICT_NAME  # 設定推文所存的 json 檔中字典的名稱
'''可修改參數'''


'''固定參數'''
CATEGORIES = ["normal", "spammer", "robot"]
'''固定參數'''


summary_data = []

for NOW_CATEGORY in CATEGORIES:
    # 所有當月的 NOW_CATEGORY 檔案
    FILTERED_JSON_FOLDER = f'../data/filtered_tweets/{NOW_CATEGORY}_tweets/{COIN_YEAR}/{COIN_MONTH}'  # JSON 檔案所在的資料夾路徑
    
    json_filenames = [f for f in os.listdir(FILTERED_JSON_FOLDER) if f.startswith(COIN_SHORT_NAME) and f.endswith(f"_{NOW_CATEGORY}.json")]

    for filename in sorted(json_filenames):
        if filename.startswith("DOGE_20250313_Latest282_Top178"):
            continue

        print(f"{NOW_CATEGORY.upper()} | {filename}:")

        # 取得日期字串：DOGE_20250320_filtered.json → 2025-03-20
        date_part = filename.split("_")[1]
        date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

        path_filtered = os.path.join(FILTERED_JSON_FOLDER, filename)
        with open(path_filtered, 'r', encoding='utf-8-sig') as f:
            data_json = json.load(f)

        # 是否為空清單
        if not data_json.get(JSON_DICT_NAME):  # 若 dogecoin 不存在或是空清單
            print(f"⚠️ 無資料或格式異常：{filename}")
            continue

        count = 0
        tweetCount = 0
        lastHourCount = 0  # 紀錄到上一個小時為止的所有推文數量
        last_count = data_json[JSON_DICT_NAME][-1]['tweet_count']
        tweetCount_perHour = []  # 記錄每小時的推文數量
        hour = 23  # 從 23 hr 開始往 0 hr 倒推

        try:
            while True:
                current_hour = int(data_json[JSON_DICT_NAME][count]['created_at'][11:13])  # 取 "HH"
                if current_hour == hour:
                    tweetCount = data_json[JSON_DICT_NAME][count]['tweet_count']
                    count += 1
                else:
                    tweetCount_perHour.append(tweetCount - lastHourCount)  # 每小時 tweet 數
                    # tweetCount_perHour.append(tweetCount)  # 每個小時累加的數量
                    hour -= 1
                    lastHourCount = tweetCount

                if data_json[JSON_DICT_NAME][count]['tweet_count'] == last_count:
                    tweetCount_perHour.append(tweetCount - lastHourCount + 1)  # 補最後一筆
                    # tweetCount_perHour.append(last_count)  # 每個小時累加的數量
                    break
        except IndexError:
            print(f"⚠️ 資料格式不足或錯誤：{filename}")
            continue

        for i in range(len(tweetCount_perHour)):
            h = 23 - i
            c = tweetCount_perHour[i]
            summary_data.append({
                "category": NOW_CATEGORY,
                "date": date_str,
                "hour": h,
                "count": c
            })


# 輸出 CSV（轉置格式：橫向為日期，縱向為小時）
df = pd.DataFrame(summary_data)

# 使用三層索引（category, hour）→ 欄位為 date
df = df.pivot_table(index=["category", "hour"], columns="date", values="count")

# 排序並重設索引
df = df.sort_index(level=["category", "hour"]).reset_index()

# 日期欄位（排除 category 與 hour）
date_cols = [col for col in df.columns if col not in ['category', 'hour']]

# 建立累積表 df_cumsum（只對日期欄做 cumsum）
df_cumsum = df.copy()
df_cumsum[date_cols] = 0  # 初始化

# 分類別計算累積推文
for cat, group in df.groupby("category"):
    cumulative = group[date_cols].cumsum()
    df_cumsum.loc[group.index, date_cols] = cumulative.values

# 將累積欄位名稱改為「累積_日期」
df_cumsum = df_cumsum.rename(columns={col: f"累積_{col}" for col in date_cols})

# 插入空欄位
df[""] = ''

# 組合原始 df + 空欄 + 累積欄
df_final = pd.concat([df, df_cumsum[[f"累積_{col}" for col in date_cols]]], axis=1)

# 插入空白列區分每個 category
dfs_with_blank = []
for cat, group_df in df_final.groupby("category"):
    dfs_with_blank.append(group_df)

    # 空白列
    blank_row = pd.DataFrame({col: '' for col in df_final.columns}, index=[0])
    dfs_with_blank.append(blank_row)

# 合併
final_df = pd.concat(dfs_with_blank, ignore_index=True)

# 輸出
output_path = f"../data/tweets/summary/{COIN_SHORT_NAME}_{COIN_YEAR}_{COIN_MONTH}_tweet_perhour.csv"
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

# len(df.columns)-2 是因為在最終輸出的 DataFrame 中，會有兩個非「日期」的欄位 (category, hour)
print(f"✅ 已儲存：{output_path}（共 {len(df.columns)-2} 天 × {len(CATEGORIES)} 類別）")  