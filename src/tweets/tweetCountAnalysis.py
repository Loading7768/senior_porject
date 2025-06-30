import os
import json
import pandas as pd
from datetime import datetime

'''可修改參數'''
COIN_YEAR = "2021"

COIN_MONTH = "06"

COIN_SHORT_NAME = "DOGE"  # 要當成檔案名的 memecoin 名稱

JSON_FOLDER = f'../data/tweets/{COIN_SHORT_NAME}/{COIN_YEAR}/{COIN_MONTH}'  # JSON 檔案所在的資料夾路徑

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱
'''可修改參數'''


'''固定參數'''
FILTERED_JSON_FOLDER = os.path.join('..', 'data', 'filtered_tweets')

NORMAL_TWEETS_FOLDER = os.path.join(FILTERED_JSON_FOLDER, 'normal_tweets', f'{COIN_YEAR}', f'{COIN_MONTH}')

SPAMMER_TWEETS_FOLDER = os.path.join(FILTERED_JSON_FOLDER, 'spammer_tweets', f'{COIN_YEAR}', f'{COIN_MONTH}')

ROBOT_TWEETS_FOLDER = os.path.join(FILTERED_JSON_FOLDER, 'robot_tweets', f'{COIN_YEAR}', f'{COIN_MONTH}')
'''固定參數'''

# 儲存最終結果
summary_data = []


# 所有原始檔案
json_filenames = [f for f in os.listdir(JSON_FOLDER) if f.startswith(COIN_SHORT_NAME) and f.endswith(".json")]

for filename in sorted(json_filenames):
    path_original = os.path.join(JSON_FOLDER, filename)
    
    # 根據原始檔名產生對應的 filtered (三種) 檔名
    base_name = filename[:-5]  # 去掉 .json  ex. DOGE_20210101
    normal_filename = f"{base_name}_normal.json"
    spammer_filename = f"{base_name}_spammer.json"
    robot_filename = f"{base_name}_robot.json"
    path_normal = os.path.join(NORMAL_TWEETS_FOLDER, normal_filename)
    path_spammer = os.path.join(SPAMMER_TWEETS_FOLDER, spammer_filename)
    path_robot = os.path.join(ROBOT_TWEETS_FOLDER, robot_filename)

    # 檢查 filtered 檔案是否存在
    if (not os.path.exists(path_normal)) or (not os.path.exists(path_spammer)) or (not os.path.exists(path_robot)):
        if not os.path.exists(path_normal):
            missing = normal_filename
            print(path_normal)
        elif not os.path.exists(path_spammer):
            missing = spammer_filename
        else:
            missing = robot_filename
        print(f"⚠️ 找不到對應的過濾檔案：{missing}")
        continue

    with open(path_original, 'r', encoding='utf-8-sig') as f:
        data_original = json.load(f)
    with open(path_normal, 'r', encoding='utf-8-sig') as f:
        data_normal = json.load(f)
    with open(path_spammer, 'r', encoding='utf-8-sig') as f:
        data_spammer = json.load(f)
    with open(path_robot, 'r', encoding='utf-8-sig') as f:
        data_robot = json.load(f)

    # 抓出原始的 tweet 總數
    tweets_original = data_original.get(JSON_DICT_NAME, [])
    tweet_total_original = len(tweets_original)

    # 抓出 normal 的 tweet 總數
    tweets_normal = data_normal.get(JSON_DICT_NAME, [])
    normal_tweet_count = len(tweets_normal)

    # 抓出 spammer 的 tweet 總數
    tweets_spammer = data_spammer.get(JSON_DICT_NAME, [])
    spammer_tweet_count = len(tweets_spammer)

    # 抓出 robot 的 tweet 總數
    tweets_robot = data_robot.get(JSON_DICT_NAME, [])
    robot_tweet_count = len(tweets_robot)

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
            "normal_tweet_count": normal_tweet_count,
            "spammer_tweet_count": spammer_tweet_count,
            "robot_tweet_count": robot_tweet_count,
            "tweet_total": tweet_total_original
        })



# 輸出 CSV
df = pd.DataFrame(summary_data)
df = df.sort_values(by="date")  # 加上這行進行排序

df.to_csv(f"../data/tweets/summary/{COIN_SHORT_NAME}_{COIN_YEAR}_{COIN_MONTH}_tweet_summary.csv", index=False)
print(f"已儲存：{COIN_SHORT_NAME}_{COIN_YEAR}_{COIN_MONTH}_tweet_summary.csv")
