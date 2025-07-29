import json
import os
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import re
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
grouped_tweets = defaultdict(list)

for json_file in tqdm(json_files, desc="抓取特定帳號的推文"):
    with open(json_file, 'r', encoding="utf-8-sig") as file:
        data = json.load(file)

    tweets = data.get(JSON_DICT_NAME, [])

    for tweet in tweets:
        username = tweet.get("username", "").lower()
        user_account = tweet.get("user_account", "").lower()
        text = tweet.get("text", "").lower()

        matched_key = None

        # 優先比對作者身份：(1) user_account  (2) username  (3) text
        if user_account != "" and (user_account in account_to_keyname):
            matched_key = account_to_keyname[user_account]
        elif username in account_to_keyname:
            matched_key = account_to_keyname[username]
        else:
            # 只針對 user_account 檢查 @提及
            for ua in user_accounts_set:
                if re.search(rf'@{re.escape(ua)}\b', text):
                    matched_key = account_to_keyname[ua]
                    break

        if matched_key:
            grouped_tweets[matched_key].append(tweet)

# 輸出每個帳號一個 json 檔案
output_dir = "../data/tweets/matched_influencer_tweets"
os.makedirs(output_dir, exist_ok=True)

for account_name, tweets in grouped_tweets.items():
    output_path = os.path.join(output_dir, f"{account_name}.json")

    # 從 1 開始標記每筆推文的 tweet_count
    for i, tweet in enumerate(tweets, start=1):
        tweet["tweet_count"] = i

    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(tweets, f, ensure_ascii=False, indent=4)

print(f"✅ 完成：共輸出 {len(grouped_tweets)} 個影響者的推文")
