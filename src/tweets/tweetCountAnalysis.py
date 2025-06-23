import os
import json
import pandas as pd
from datetime import datetime

'''可修改參數'''
COIN_YEAR = "2021"

COIN_MONTH = "08"

COIN_SHORT_NAME = "DOGE"  # 要當成檔案名的 memecoin 名稱

JSON_FOLDER = f'../data/tweets/{COIN_SHORT_NAME}/{COIN_YEAR}/{COIN_MONTH}'  # JSON 檔案所在的資料夾路徑

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱

FILTERED_JSON_FOLDER = f'../data/tweets/{COIN_SHORT_NAME}/{COIN_YEAR}/{COIN_MONTH}/filtered'  # JSON 檔案所在的資料夾路徑
'''可修改參數'''

# 儲存最終結果
summary_data = []

# 所有原始檔案（不含 _filtered）
json_filenames = [f for f in os.listdir(JSON_FOLDER) if f.startswith(COIN_SHORT_NAME) and f.endswith(".json") and "_filtered" not in f]

for filename in sorted(json_filenames):
    path_original = os.path.join(JSON_FOLDER, filename)
    
    # 根據原始檔名產生對應的 filtered 檔名
    base_name = filename[:-5]  # 去掉 .json
    filtered_filename = f"{base_name}_filtered.json"
    path_filtered = os.path.join(FILTERED_JSON_FOLDER, filtered_filename)

    # 檢查 filtered 檔案是否存在
    if not os.path.exists(path_filtered):
        print(f"⚠️ 找不到對應的過濾檔案：{filtered_filename}")
        continue

    with open(path_original, 'r', encoding='utf-8-sig') as f:
        data_original = json.load(f)
    with open(path_filtered, 'r', encoding='utf-8-sig') as f:
        data_filtered = json.load(f)

    # 抓出原始的 tweet 總數
    tweets_original = data_original.get(JSON_DICT_NAME, [])
    tweet_total_original = len(tweets_original)

    # 抓出 filtered 的 tweet 總數
    tweets_filtered = data_filtered.get(JSON_DICT_NAME, [])
    normal_tweet_count = len(tweets_filtered)

    # 抓出 spammer 的 tweet 總數
    spammer_tweet_count = tweet_total_original - normal_tweet_count

    if tweet_total_original > 0:
        # 第一筆是最晚的，最後一筆是最早的
        latest_tweet = tweets_original[0]
        earliest_tweet = tweets_original[-1]

        # 提取時間字串
        latest_str = latest_tweet.get("created_at")
        earliest_str = earliest_tweet.get("created_at")

        # 轉換成 datetime
        latest_dt = datetime.strptime(latest_str, "%a %b %d %H:%M:%S %z %Y")
        earliest_dt = datetime.strptime(earliest_str, "%a %b %d %H:%M:%S %z %Y")

        # 日期（取其中一筆就好）
        date_str = latest_dt.strftime("%Y-%m-%d")
        latest_time = latest_dt.strftime("%H:%M:%S")
        earliest_time = earliest_dt.strftime("%H:%M:%S")

        # 判斷有沒有抓完 是否是 00:XX:XX
        isCompleteData = earliest_time.startswith("00:")

        summary_data.append({
            "filename": filename,
            "date": date_str,
            "start_time": latest_time,
            "finish_time": earliest_time,
            "isCompleteData": isCompleteData,
            "spammer_tweet_count": spammer_tweet_count,
            "normal_tweet_count": normal_tweet_count,
            "tweet_total": tweet_total_original
        })



# 輸出 CSV
df = pd.DataFrame(summary_data)
df = df.sort_values(by="date")  # 加上這行進行排序

df.to_csv(f"../data/tweets/summary/{COIN_SHORT_NAME}_{COIN_YEAR}_{COIN_MONTH}_tweet_summary.csv", index=False)
print(f"已儲存：{COIN_SHORT_NAME}_{COIN_YEAR}_{COIN_MONTH}_tweet_summary.csv")
