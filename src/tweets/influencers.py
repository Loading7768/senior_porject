import json
import os
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import re
from datetime import datetime
from pathlib import Path


# 匯入 config
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

# 所有推文 JSON 檔案
json_files = glob(f'../data/tweets/{COIN_SHORT_NAME}/*/*/*.json')

# 讀取影響者清單
influencers_path = "../data/tweets/influencers.json"
with open(influencers_path, 'r', encoding="utf-8-sig") as file:
    influencers_list = json.load(file)

# 建立：user_account 對應成統一 key（作為輸出檔名）
account_to_keyname = {}
user_accounts_set = set()
for inf in influencers_list:
    standard_name = inf["user_account"]  # 決定輸出檔名
    username = inf["username"].lower()
    user_account = inf["user_account"].lower()
    
    account_to_keyname[username] = standard_name
    account_to_keyname[user_account] = standard_name
    user_accounts_set.add(user_account)

# 所有符合的推文，依帳號分開
grouped_tweets = defaultdict(list)           # 所有相關推文（包含被提及）
author_tweets = defaultdict(list)            # 僅作者本人推文

for json_file in tqdm(json_files, desc="抓取特定帳號的推文"):
    with open(json_file, 'r', encoding="utf-8-sig") as file:
        data = json.load(file)

    tweets = data.get(JSON_DICT_NAME, [])

    for tweet in tweets:
        username = tweet.get("username", "").lower()
        user_account = tweet.get("user_account", "").lower()
        text = tweet.get("text", "").lower()

        matched_key = None
        match_type = None

        # 優先比對作者身份：(1) user_account  (2) username  (3) text
        if user_account != "" and (user_account in account_to_keyname):
            matched_key = account_to_keyname[user_account]
            match_type = "user_account"
            grouped_tweets[matched_key].append(tweet)
            author_tweets[matched_key].append(tweet)

        elif username in account_to_keyname:
            matched_key = account_to_keyname[username]
            match_type = "username"
            grouped_tweets[matched_key].append(tweet)
            author_tweets[matched_key].append(tweet)

        else:
            # 只針對 user_account 檢查 @提及
            for ua in user_accounts_set:
                if re.search(rf'@{re.escape(ua)}\b', text):
                    matched_key = account_to_keyname[ua]
                    match_type = "text"
                    grouped_tweets[matched_key].append(tweet)
                    break

        if matched_key:
            tweet["match_type"] = match_type  # 標記匹配來源
            # grouped_tweets[matched_key].append(tweet)

# 輸出每個帳號一個 json 檔案
output_dir = "../data/tweets/matched_influencer_tweets"
os.makedirs(output_dir, exist_ok=True)

match_priority = {"user_account": 0, "username": 1, "text": 2}

for account_name, tweets in grouped_tweets.items():
    output_path = os.path.join(output_dir, f"{account_name}.json")

    # 依照 match_type 排序
    tweets.sort(key=lambda tweet: match_priority.get(tweet.get("match_type", "text")))

    # 從 1 開始標記 tweet_count
    for i, tweet in enumerate(tweets, start=1):
        tweet["tweet_count"] = i


    # === Spammer 判斷（滑動視窗3600秒） ===

    # 只拿作者本人推文來做 spammer 判斷
    author_list = author_tweets.get(account_name, [])

    # 把作者本人推文依時間排序（你可能也想做 tweet_count 這步驟）
    author_list.sort(key=lambda tweet: tweet.get("created_at"))

    is_spammer = False
    times = []

    # 先把所有推文時間轉成 timestamp（秒）
    for tweet in author_list:
        created_time = tweet.get("created_at")
        if created_time:
            # 依實際格式調整解析
            dt = datetime.strptime(created_time, "%a %b %d %H:%M:%S %z %Y")
            times.append(dt.timestamp())

    times.sort()

    start = 0  # 視窗左邊界
    for end in range(len(times)):  # end: 視窗右邊界
        # 保持 times[end] - times[start] <= 3600
        while times[end] - times[start] > 3600:
            start += 1
        if (end - start + 1) >= 6:  # 至少 6 篇
            is_spammer = True
            break

    # === 如果是 spammer，額外存檔 ===
    output_spammer_path = os.path.join(output_dir, f"{account_name}_spammer.json")
    if is_spammer and author_list:
        for i, tweet in enumerate(author_list, start=1):
            tweet["tweet_count"] = i

        with open(output_spammer_path, "w", encoding="utf-8-sig") as f:
            json.dump(author_list, f, ensure_ascii=False, indent=4)

        print(f"💾 已儲存 spammer 作者推文: {output_spammer_path}")


    # 儲存 spammer 標記
    output_data = {
        "account_name": account_name,
        "is_spammer": is_spammer,
        "tweets": tweets
    }

    # 輸出每個帳號所有的推文
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"{account_name}.json: 共 {len(tweets)} 筆 | spammer: {is_spammer}")

print(f"✅ 完成：共輸出 {len(grouped_tweets)} 個影響者的推文")
