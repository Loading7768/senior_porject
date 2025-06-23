from pathlib import Path
import json
from datetime import datetime

# === 可自由調整的參數 ===
COIN = "PEPE"          # 幣種（檔名裡的代號）
YEAR = "2025"             # 西元年（int）
MONTH = "03"               # 月份（int）→ 會自動補 0
AUTHOR_LIST_FILE = f"../data/dice/{COIN}/robot_list/{COIN}_{YEAR}{MONTH}_list.txt"   # 黑名單作者清單
DATA_DIR = Path(f"../data/tweets/{COIN}/{YEAR}/{MONTH}")                   # 原始 JSON 資料夾
OUT_DIR = Path(f"../data/filtered_tweets/{COIN}/{YEAR}/{MONTH}")                                               # 輸出結果存這裡

# === 準備輸出資料夾 ===
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 讀黑名單作者 ===
with open(AUTHOR_LIST_FILE, encoding="utf-8") as f:
    blocked_users = {line.strip() for line in f if line.strip()}

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

total_removed = 0
for fp in files:
    # ✅ 新增檢查已處理的檔案
    filtered_fp = OUT_DIR / fp.name.replace(".json", "_filtered.json")
    if filtered_fp.exists():
        print(f"跳過 {fp.name}：已存在 {filtered_fp.name}")
        continue

    # 讀檔（處理可能的 BOM）
    with fp.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    tweets = data.get(COIN, [])
    # 先過濾
    filtered = [tw for tw in tweets if tw.get("username") not in blocked_users]

    # 依 created_at 時間排序（預設由舊到新）
    filtered.sort(key=lambda tw: parse_time(tw.get("created_at", "")))

    # 重新編號 tweet_count，確保順序正確
    for idx, tw in enumerate(filtered, start=1):
        tw["tweet_count"] = idx

    removed = len(tweets) - len(filtered)
    total_removed += removed

    # 輸出
    with filtered_fp.open("w", encoding="utf-8") as f:
        json.dump({COIN: filtered}, f, ensure_ascii=False, indent=4)

    print(f"{fp.name}: 去掉 {removed} 則，剩 {len(filtered)} 則 → {filtered_fp.name}")

print(f"\n整個 {YEAR}-{MONTH} 共移除 {total_removed} 則推文")
