import os
import json
import csv
from collections import Counter

def analyze_tweets(folder_path, coin_type, output_csv):
    daily_author_stats = []
    all_post_counts = set()

    json_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".json")
    ])

    for filename in json_files:
        date_str = filename.replace(f"{coin_type}_", "").replace(".json", "")
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            if coin_type not in data:
                continue

            tweets = data[coin_type]
            tweet_count = len(tweets)
            author_counter = Counter(tweet["username"] for tweet in tweets)

            unique_authors = len(author_counter)
            multiple_post_authors = sum(1 for c in author_counter.values() if c > 1)

            post_count_distribution = Counter(author_counter.values())
            all_post_counts.update(post_count_distribution.keys())

            row = {
                "Date": date_formatted,
                "Unique Authors": unique_authors,
                "Tweet Count": tweet_count,
                "Authors With Multiple Posts": multiple_post_authors
            }

            for count, user_num in post_count_distribution.items():
                row[f"Authors Posting {count} Time{'s' if count > 1 else ''}"] = user_num

            daily_author_stats.append(row)

        except Exception as e:
            print(f"無法處理 {filename}：{e}")

    fixed_columns = ["Date", "Unique Authors", "Tweet Count", "Authors With Multiple Posts"]
    dynamic_columns = [f"Authors Posting {i} Time{'s' if i > 1 else ''}" for i in sorted(all_post_counts)]
    all_columns = fixed_columns + dynamic_columns

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in daily_author_stats:
            for col in dynamic_columns:
                row.setdefault(col, 0)
            writer.writerow(row)

    print(f"✅ 統計完成：{output_csv}")


# ✅ 使用範例（直接執行）
if __name__ == "__main__":
    folder = "data/PEPE/2025"       # 📂 資料夾路徑
    coin = "PEPE"                   # 🪙 幣種名稱（對應 JSON 裡的 key）
    output = "pepe_post_stats.csv"  # 📄 匯出檔名

    analyze_tweets(folder, coin, output)
