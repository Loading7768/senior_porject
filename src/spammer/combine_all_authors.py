import os
import json
import re
from collections import defaultdict
from pathlib import Path

# === 自訂參數 ===
JSON_COIN = "PEPE"
COIN_TYPE = "PEPE"
YEAR = "2024"
MONTH = "11"

# === 資料夾設定 ===
folder_path = f"../data/tweets/{COIN_TYPE}/{YEAR}/{MONTH}"
file_prefix = f"{COIN_TYPE}_{YEAR}{MONTH}"
output_folder_path = f"../data/author_all/{COIN_TYPE}/{YEAR}/{MONTH}/"
os.makedirs(output_folder_path, exist_ok=True)


def sanitize_filename(name):
    """將非法檔名字元替換掉"""
    return re.sub(r'[\\/*?:"<>|]', '_', name)


# === 儲存所有作者的推文 ===
all_author_tweets = defaultdict(lambda: {JSON_COIN: []})  # {author: {"PEPE": [tweets]}}

# === 掃描所有 JSON 推文檔 ===
for filename in sorted(os.listdir(folder_path)):
    if filename.startswith(file_prefix) and filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 讀取失敗：{filename}，錯誤：{e}")
            continue

        if JSON_COIN not in data:
            continue

        for tweet in data[JSON_COIN]:
            username = tweet.get("user_account") or tweet.get("username")
            all_author_tweets[username][JSON_COIN].append(tweet)

# === 輸出每位作者的所有推文為 JSON 檔（僅保留發文 > 1 的） ===
for author, tweet_dict in all_author_tweets.items():
    tweet_count = len(tweet_dict[JSON_COIN])
    if tweet_count <= 1:
        continue  # ❌ 跳過只發一篇的作者

    safe_name = sanitize_filename(author)
    output_path = os.path.join(output_folder_path, f"{safe_name}_{YEAR}{MONTH}.json")

    try:
        with open(output_path, "w", encoding="utf-8-sig") as f:
            json.dump(tweet_dict, f, indent=4, ensure_ascii=False)
        print(f"✅ 已儲存 {output_path}（共 {tweet_count} 篇）")
    except Exception as e:
        print(f"❌ 儲存失敗：{author} -> {e}")
