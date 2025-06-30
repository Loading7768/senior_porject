import os
import json
from pathlib import Path
import re

# === 自訂參數 ===
JSON_COIN = "dogecoin"
COIN_TYPE = "DOGE"
YEAR = "2021"
MONTH = "05"


def sanitize_filename(name):
    # 移除或替換不合法的檔名字元
    return re.sub(r'[\\/*?:"<>|]', '_', name)


# === 資料夾設定 ===
folder_path = f"../data/tweets/{COIN_TYPE}/{YEAR}/{MONTH}"
# folder_path = f"data/tweets"
file_prefix = f"{COIN_TYPE}_{YEAR}{MONTH}"
output_folder_path = f"../data/spammer/{YEAR}/{MONTH}/"
os.makedirs(output_folder_path, exist_ok=True)

spammer_list_path = f"../data/spammer/{YEAR}/spammer_{YEAR}{MONTH}.txt"
with open(spammer_list_path, "r", encoding="utf-8-sig") as f:
    target_users = [line.strip() for line in f if line.strip()]

with open(f"../data/spammer/{YEAR}/spammer_{YEAR}{MONTH}_log.txt", "w", encoding="utf-8-sig") as file:
    file.write("")

# === 掃描所有檔案 ===
for target_user in sorted(target_users):
    filtered_tweets = {JSON_COIN: []}

    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith(file_prefix) and filename.endswith(".json"):
            date_str = filename.replace(f"{COIN_TYPE}_", "").replace(".json", "")
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                # print(f"讀取 {filename} 時失敗：{e}")
                continue

            if JSON_COIN not in data:
                continue
            
            # 如果 user_account 或 username 存在在 target_user 就把那篇 tweet 存到 filtered_tweets
            for tweet in data[JSON_COIN]:
                if tweet.get("user_account") == target_user or tweet.get("username") == target_user:
                    filtered_tweets[JSON_COIN].append(tweet)

            # filtered_tweets[json_coin].extend(tweet for tweet in data[json_coin] if tweet["username"] == target_user)

    count = 0
    last_text = filtered_tweets[JSON_COIN][-1]['created_at']
    while True:
        number = filtered_tweets[JSON_COIN][count]['tweet_count']
        filtered_tweets[JSON_COIN][count]['tweet_count'] = count + 1
        
        if filtered_tweets[JSON_COIN][count]['created_at'] == last_text:
            break

        count += 1

    # 在儲存前先清洗檔名
    safe_filename = sanitize_filename(target_user)

    try:
        with open(f"{output_folder_path}{target_user}_{YEAR}{MONTH}.json", "w", encoding="utf-8-sig") as file:
            json.dump(filtered_tweets, file, indent=4, ensure_ascii=False)
        status = "成功"
    except Exception as e:
        with open(f"{output_folder_path}{safe_filename}_{YEAR}{MONTH}.json", "w", encoding="utf-8-sig") as file:
            json.dump(filtered_tweets, file, indent=4, ensure_ascii=False)
        status = f"失敗: {e}"

    # 不論成功或失敗，都記錄下來
    with open(f"../data/spammer/{YEAR}/spammer_{YEAR}{MONTH}_log.txt", "a", encoding="utf-8-sig") as file:
        file.write(f"{target_user} → {safe_filename}_{YEAR}{MONTH}.json : {status}\n")