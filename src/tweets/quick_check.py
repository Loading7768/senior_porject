import os
import json
from collections import defaultdict
from datetime import datetime

# === 自訂參數 ===
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME,JSON_DICT_NAME

# === 資料夾設定 ===
root_folder = f"../data/tweets/{COIN_SHORT_NAME}"   # 掃描整個 PEPE 資料夾
output_path = f"../data/{COIN_SHORT_NAME}_daily_stats.json"  # 輸出統計檔

# === 統計每天推文數量 ===
date_count = defaultdict(int)

# === 遞迴掃描所有 JSON 推文檔 ===
for dirpath, _, filenames in os.walk(root_folder):
    for filename in sorted(filenames):
        if filename.endswith(".json") and filename.startswith(COIN_SHORT_NAME):
            file_path = os.path.join(dirpath, filename)

            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ 讀取失敗：{filename}，錯誤：{e}")
                continue

            if JSON_DICT_NAME not in data:
                continue

            for tweet in data[JSON_DICT_NAME]:
                created_at = tweet.get("created_at")
                if created_at:
                    try:
                        # 假設 created_at 格式為 "Thu Feb 01 23:59:54 +0000 2024"
                        dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                        date_str = dt.strftime("%Y-%m-%d")
                        date_count[date_str] += 1
                    except Exception:
                        pass

# === 排序後輸出到 JSON 檔 ===
sorted_stats = dict(sorted(date_count.items()))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sorted_stats, f, indent=4, ensure_ascii=False)

print(f"✅ 已輸出每日統計到 {output_path}")
