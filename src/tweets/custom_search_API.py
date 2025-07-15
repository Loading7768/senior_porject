'''
前置作業：

    申請 API 金鑰：
    → https://console.cloud.google.com/apis/credentials

    建立自訂搜尋引擎（Custom Search Engine, CSE）：
    → https://programmablesearchengine.google.com/

    如果要搜尋全網，設定「搜尋整個網路」。

    啟用 Custom Search API：
    → https://console.cloud.google.com/apis/library/customsearch.googleapis.com
'''


import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import config

import requests
from datetime import datetime, timedelta
import json
import os



'''可修改參數'''
API_KEY = ""

CX = ""

COIN_NAME = config.COIN_NAME[1:-1]

COIN_SHORT_NAME = config.COIN_SHORT_NAME

SITE = "twitter.com"

START_DATE = datetime(2025, 6, 1)

DAY_COUNT = 10  # 設定要連續抓幾天
'''可修改參數'''



def google_custom_search(query, date):
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "lr": "lang_en",  # 設定為英文頁面
        "num": 10  # 看要抓出多少篇內容 (最多 10 個)
        # "dateRestrict": "d1",  # 只看當日
    }

    print(f"查詢日期：{date} → {query}")

    # 請求 API
    response = requests.get(url, params=params) 
    if response.status_code != 200:
        print("錯誤：", response.status_code, response.text)
        return

    data = response.json()

    # 列出總搜尋數量
    total_results = data.get("searchInformation", {}).get("totalResults", "0")
    print(f"搜尋結果數量：{total_results}\n")

    # 列出前 10 項內容
    if "items" in data:
        for i, item in enumerate(data["items"], 1):
            print(f"{i}. {item.get('title')}")
            print(item.get("link"))
            print("---")
    else:
        print("沒有找到搜尋結果。")

    return int(total_results)

# === 主流程 ===
results = []

file_path = "../data/tweets/count/chrome/"
os.makedirs(file_path, exist_ok=True)
json_path = f"{file_path}{COIN_SHORT_NAME}_google_search_counts.json"

for i in range(DAY_COUNT):
    current_date = START_DATE + timedelta(days=i)
    next_date = current_date + timedelta(days=1)

    date_str = current_date.strftime("%Y-%m-%d")
    next_date_str = next_date.strftime("%Y-%m-%d")

    # 設定 QUREY (包含 日期)
    query = f"{COIN_NAME} site:{SITE} after:{date_str} before:{next_date_str}"
    # inurl:/status/    

    count = google_custom_search(query, date_str)


    results.append({
        "date": date_str,
        "count": count,
        "query": COIN_NAME,
        "site": SITE
    })


    # 先讀取之前的資料 若無則不用
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
    except:
        data_json = []
    
    data_json.append(results[-1])  # 只加當天那筆
    data_json.sort(key=lambda x: x["date"])  # 按日期排序
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=4)

    print(f"✅ 所有資料已儲存到 {file_path}{COIN_SHORT_NAME}_google_search_counts.json")

    # wait_time = 5
    # print(f"⏳ 等待 {wait_time:.2f} 秒模擬真人...")
    # time.sleep(wait_time)