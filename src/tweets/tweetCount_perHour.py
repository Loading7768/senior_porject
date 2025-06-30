import json
import os
import pandas as pd

'''可修改參數'''
COIN_YEAR = "2025"

COIN_MONTH = "02"

COIN_SHORT_NAME = "PEPE"  # 要當成檔案名的 memecoin 名稱

FILTERED_JSON_FOLDER = f'../data/filtered_tweets/{COIN_SHORT_NAME}/{COIN_YEAR}/{COIN_MONTH}'  # JSON 檔案所在的資料夾路徑

JSON_DICT_NAME = "PEPE"  # 設定推文所存的 json 檔中字典的名稱
'''可修改參數'''

summary_data = []

# 所有當月的 filtered 檔案
json_filenames = [f for f in os.listdir(FILTERED_JSON_FOLDER) if f.startswith(COIN_SHORT_NAME) and f.endswith("_filtered.json")]

for filename in sorted(json_filenames):
    if filename == "DOGE_20250313_Latest282_Top178_filtered.json":
        continue

    print(f"{filename}:")

    # 取得日期字串：DOGE_20250320_filtered.json → 2025-03-20
    date_part = filename.split("_")[1]
    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

    path_filtered = os.path.join(FILTERED_JSON_FOLDER, filename)
    with open(path_filtered, 'r', encoding='utf-8-sig') as f:
        data_json = json.load(f)

    count = 0
    tweetCount = 0
    lastHourCount = 0  # 紀錄到上一個小時為止的所有推文數量
    last_count = data_json[JSON_DICT_NAME][-1]['tweet_count']
    tweetCount_perHour = []  # 記錄每小時的推文數量
    hour = 23  # 從 23 hr 開始往 0 hr 倒推

    while True:
        current_hour = int(data_json[JSON_DICT_NAME][count]['created_at'][11:13])  # 取 "HH"
        if current_hour == hour:
            tweetCount = data_json[JSON_DICT_NAME][count]['tweet_count']
            count += 1
        else:
            tweetCount_perHour.append(tweetCount - lastHourCount)  # 每小時 tweet 數
            hour -= 1
            lastHourCount = tweetCount

        if data_json[JSON_DICT_NAME][count]['tweet_count'] == last_count:
            tweetCount_perHour.append(tweetCount - lastHourCount + 1)  # 補最後一筆
            break

    for i in range(len(tweetCount_perHour)):
        h = 23 - i
        c = tweetCount_perHour[i]
        summary_data.append({
            "date": date_str,
            "hour": h,
            "count": c
        })
        # print(f"{h} hr - {c}")
    
    # print()

# 輸出 CSV（轉置格式：橫向為日期，縱向為小時）
df = pd.DataFrame(summary_data)
df = df.pivot(index="hour", columns="date", values="count")

# 重新排序索引
df = df.reindex(list(range(0, 24)))

# 輸出
output_path = f"../data/tweets/summary/{COIN_SHORT_NAME}_{COIN_YEAR}_{COIN_MONTH}_tweet_perhour.csv"
df.to_csv(output_path, encoding='utf-8-sig')

print(f"✅ 已儲存：{output_path}（共 {len(df.columns)} 天）")