'''
separate tweets into 3 different categories
1. Normal tweets: likely from a normal twitter user
2. Spammer tweets: frequent poster with distinct contents
3. Robot tweets: frequent poster with similar contents
'''

from pathlib import Path
import json
from datetime import datetime
import os

import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import config

# === 可自由調整的參數 ===
COIN = config.COIN_SHORT_NAME          # 幣種（檔名裡的代號）
YEAR = "2025"             # 西元年（int）
MONTH = "04"               # 月份（int）→ 會自動補 0
JSON_DICT_NAME = config.JSON_DICT_NAME
AUTHOR_LIST_FILE = f"../data/dice/robot_list/{YEAR}{MONTH}_list.txt"   # 黑名單作者清單
FREQUENT_POST_LIST_FILE = f'../data/spammer/spammer_{YEAR}{MONTH}.txt'
DATA_DIR = Path(f"../data/tweets/{YEAR}/{MONTH}")                   # 原始 JSON 資料夾   
OUT_PARENT_PATH = '../data/filtered_tweets'                                         # 輸出結果存這裡

# === 準備輸出資料夾 ===
os.makedirs(f'{OUT_PARENT_PATH}/normal_tweets/{YEAR}/{MONTH}', exist_ok=True)
os.makedirs(f'{OUT_PARENT_PATH}/spammer_tweets/{YEAR}/{MONTH}', exist_ok=True)
os.makedirs(f'{OUT_PARENT_PATH}/robot_tweets/{YEAR}/{MONTH}', exist_ok=True)

# === Create set for different categories ===
with open(FREQUENT_POST_LIST_FILE, encoding="utf-8") as f:
    frequent_poster = {line.strip() for line in f if line.strip()}

with open(AUTHOR_LIST_FILE, encoding="utf-8") as f:
    robots = {line.strip() for line in f if line.strip()}

spammers = frequent_poster - robots

# === 找出當月所有 JSON 檔 ===
pattern = f"{COIN}_{YEAR}{MONTH}*.json"
files = sorted(DATA_DIR.glob(pattern))
if not files:
    raise FileNotFoundError(f"找不到符合 {pattern} 的檔案")

# === 輔助：解析 created_at 為 datetime ===

def parse_time(ts: str) -> datetime:
    """將各種常見格式的時間字串轉成 datetime；解析失敗回傳 datetime.min。"""
    if not ts:
        return datetime.min
    # ISO 8601 帶 Z → 改成 +00:00
    ts = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # 備援：嘗試常見格式 2025-02-15 12:34:56
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.min

# === 主迴圈 ===
for fp in files:
    # ✅ 新增檢查已處理的檔案
    # filtered_fp = OUT_DIR / fp.name.replace(".json", "_filtered.json")
    # if filtered_fp.exists():
    #     print(f"跳過 {fp.name}：已存在 {filtered_fp.name}")
    #     continue

    # 讀檔（處理可能的 BOM）
    with fp.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    tweets = data.get(JSON_DICT_NAME, [])
    # 先過濾
    normal_tweets = [t for t in tweets if t.get('username') not in frequent_poster]
    spammer_tweets = [t for t in tweets if t.get('username') in spammers]
    robot_tweets = [t for t in tweets if t.get("username") in robots]

    # 依 created_at 時間排序（預設由舊到新）
    # filtered.sort(key=lambda tw: parse_time(tw.get("created_at", "")))

    # 重新編號 tweet_count，確保順序正確
    for idx, t in enumerate(normal_tweets, start=1):
        t["tweet_count"] = idx

    for idx, t in enumerate(spammer_tweets, start=1):
        t["tweet_count"] = idx

    for idx, t in enumerate(robot_tweets, start=1):
        t["tweet_count"] = idx


    # 輸出
    fp_normal = Path(OUT_PARENT_PATH) / "normal_tweets" / YEAR / MONTH / (fp.stem + "_normal.json")
    with fp_normal.open("w", encoding="utf-8") as f:
        json.dump({JSON_DICT_NAME: normal_tweets}, f, ensure_ascii=False, indent=4)

    fp_spammer = Path(OUT_PARENT_PATH) / "spammer_tweets" / YEAR / MONTH / (fp.stem + "_spammer.json")
    with fp_spammer.open("w", encoding="utf-8") as f:
        json.dump({JSON_DICT_NAME: spammer_tweets}, f, ensure_ascii=False, indent=4)

    fp_robot = Path(OUT_PARENT_PATH) / "robot_tweets" / YEAR / MONTH / (fp.stem + "_robot.json")
    with fp_robot.open("w", encoding="utf-8") as f:
        json.dump({JSON_DICT_NAME: robot_tweets}, f, ensure_ascii=False, indent=4)
