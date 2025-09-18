import os
import json
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from pepe_config import JSON_DICT_NAME, COIN_SHORT_NAME

# === 自訂參數 ===
LIMIT = 20  # 每天超過這個數量的推文就標記為 spammer

# === 路徑 ===
root_folder = f"../data/author_all/{COIN_SHORT_NAME}"
output_folder = f"../data/spammer/{COIN_SHORT_NAME}/"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, f"{COIN_SHORT_NAME}_spammers.txt")

# === 解析時間（兼容幾種常見格式） ===
def parse_time(s):
    if not s:
        return None
    for fmt in ("%a %b %d %H:%M:%S %z %Y",  # Mon Jan 01 12:34:56 +0000 2024
                "%a %b %d %H:%M:%S %Y",    # Mon Jan 01 12:34:56 2024
                "%Y-%m-%d %H:%M:%S%z",     # 2024-01-01 12:34:56+0000
                "%Y-%m-%d %H:%M:%S"):      # 2024-01-01 12:34:56
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

# === 收集每位作者的所有發文時間（跨所有檔案） ===
author_times = defaultdict(list)

# 先統計所有檔案數量，方便 tqdm 顯示
all_files = []
for dirpath, _, filenames in os.walk(root_folder):
    for filename in sorted(filenames):
        if filename.endswith(".json"):
            all_files.append(os.path.join(dirpath, filename))

print(f"🔍 準備處理 {len(all_files)} 個檔案...")

for file_path in tqdm(all_files, desc="讀取 JSON 檔案"):
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 讀取失敗：{file_path}，錯誤：{e}")
        continue

    tweets = data.get(JSON_DICT_NAME)
    if not isinstance(tweets, list):
        continue

    for tw in tweets:
        user = tw.get("user_account") or tw.get("username")
        t = parse_time(tw.get("created_at", ""))
        if user and t:
            author_times[user].append(t)

# === 判斷條件：某一天內推文數量超過 LIMIT 就是 spammer ===
spammers = set()

print(f"⚡ 開始檢查 {len(author_times)} 位作者...")

for user, times in tqdm(author_times.items(), desc="檢查作者"):
    if len(times) < LIMIT:
        continue

    # 將時間轉為 date（只保留年月日）並統計每一天的推文數量
    date_counts = defaultdict(int)
    for t in times:
        date_counts[t.date()] += 1

    # 如果任何一天超過 LIMIT 則視為 spammer
    if any(count >= LIMIT for count in date_counts.values()):
        spammers.add(user)

# === 輸出 spammer 名單 ===
with open(output_file, "w", encoding="utf-8-sig") as f:
    for user in sorted(spammers):
        f.write(user + "\n")

print(f"✅ 已完成檢測：共 {len(spammers)} 位 spammer，名單已存入：{output_file}")
