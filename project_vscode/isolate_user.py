import os
import json
from pathlib import Path

# === 自訂參數 ===
json_coin = "(officialtrump OR \"official trump\" OR \"trump meme coin\" OR \"trump coin\" OR trumpcoin OR $TRUMP OR \"dollar trump\")"
coin_type = "TRUMP"
year = "2025"
month = "03"

# === 資料夾設定 ===
# folder_path = f"data/{coin_type}/{year}/{month}"
folder_path = f"data/tweets"
file_prefix = f"{coin_type}_{year}{month}"
output_folder_path = f"data/spammer/{month}/"
os.makedirs(output_folder_path, exist_ok=True)

spammer_list_path = f"data/spammer/spammer_{month}.txt"
with open(spammer_list_path, "r") as f:
    target_users = [line.strip() for line in f if line.strip()]

# === 掃描所有檔案 ===
for target_user in target_users:
    filtered_tweets = {json_coin: []}

    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith(file_prefix) and filename.endswith(".json"):
            date_str = filename.replace(f"{coin_type}_", "").replace(".json", "")
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                # print(f"讀取 {filename} 時失敗：{e}")
                continue

            if json_coin not in data:
                continue

            filtered_tweets[json_coin].extend(tweet for tweet in data[json_coin] if tweet["username"] == target_user)

    count = 0
    last_text = filtered_tweets[json_coin][-1]['created_at']
    while True:
        number = filtered_tweets[json_coin][count]['tweet_count']
        filtered_tweets[json_coin][count]['tweet_count'] = count + 1
        
        if filtered_tweets[json_coin][count]['created_at'] == last_text:
            break

        count += 1

    with open(f"{output_folder_path}{target_user}_{year}{month}.json", "w") as file:
        json.dump(filtered_tweets, file, indent=4)
