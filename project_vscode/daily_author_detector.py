import json
import glob
import os
import re
from datetime import datetime
from collections import defaultdict

# 移除檔名中的非法字元
def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', "_", name)

def find_active_authors_and_save(coin, year, month):
    # 資料夾與檔案設定
    folder_path = os.path.join("data", coin, str(year), f"{month:02d}")
    file_pattern = f"{coin}_{year}{month:02d}*.json"
    file_paths = sorted(glob.glob(os.path.join(folder_path, file_pattern)))

    author_dates = defaultdict(set)
    author_tweets = defaultdict(list)
    burst_authors = set()  # 有爆量發文行為的作者

    # 逐檔處理
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
                tweets = data.get("(officialtrump OR \"official trump\" OR \"trump meme coin\" OR \"trump coin\" OR trumpcoin OR $TRUMP OR \"dollar trump\")", [])

                user_tweets = defaultdict(list)  # 單日作者發文時間

                for tweet in tweets:
                    username = tweet["username"]
                    dt = datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S %z %Y")
                    date_str = dt.strftime("%Y-%m-%d")

                    author_dates[username].add(date_str)
                    author_tweets[username].append(tweet)
                    user_tweets[username].append(dt)

                # 判斷是否有 1 小時內發超過 6 篇
                for user, times in user_tweets.items():
                    if len(times) <= 1:
                        continue
                    times.sort()
                    for i in range(len(times)):
                        count = 1
                        for j in range(i + 1, len(times)):
                            if (times[j] - times[i]).total_seconds() <= 3600:
                                count += 1
                                if count >= 6:
                                    burst_authors.add(user)
                                    break
                            else:
                                break
                        if count >= 6:
                            break
        except Exception as e:
            print(f"❌ 無法讀取 {file_path}：{e}")

    qualified_authors = set()

    # 篩選：發文天數 ≥10 且出現連續 3 天發文
    for author, date_strs in author_dates.items():
        sorted_dates = sorted(datetime.strptime(d, "%Y-%m-%d") for d in date_strs)
        if len(sorted_dates) < 10:
            continue
        for i in range(len(sorted_dates) - 2):
            if (sorted_dates[i+1] - sorted_dates[i]).days == 1 and (sorted_dates[i+2] - sorted_dates[i+1]).days == 1:
                qualified_authors.add(author)
                break

    # 加入爆量作者
    qualified_authors.update(burst_authors)

    # 建立 spammers 資料夾
    spammers_folder = os.path.join(folder_path, "spammers")
    os.makedirs(spammers_folder, exist_ok=True)

    # 輸出每位作者的 json
    for author in qualified_authors:
        safe_author = sanitize_filename(author)
        author_file = os.path.join(spammers_folder, f"{safe_author}_{year}{month:02d}.json")

        tweets = author_tweets[author]
        for idx, tweet in enumerate(tweets, start=1):
            tweet["tweet_count"] = idx

        author_data = {
            coin: tweets
        }

        with open(author_file, "w", encoding="utf-8") as f:
            json.dump(author_data, f, ensure_ascii=False, indent=2)

    return qualified_authors

# ✅ 執行程式
if __name__ == "__main__":
    coin = "TRUMP"
    year = 2025
    month = 3

    result = find_active_authors_and_save(coin, year, month)
    print(f"\n✅ 共找到 {len(result)} 位疑似 spammer 作者：")
    #for author in result:
        #print("-", author)
