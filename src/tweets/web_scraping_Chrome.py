import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import time
import random
import os
import json
import re

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import config


'''可修改參數'''
COIN_NAME = config.GOOGLE_QUERY

COIN_SHORT_NAME = config.COIN_SHORT_NAME

SITE = "twitter.com"

START_DATE = datetime(2025, 7, 1)  # 查詢開始日期

DAY_COUNT = 10  # 要連續找幾天

QUERY = f"{COIN_NAME} site:{SITE}"
'''可修改參數'''

results = []  # 儲存所有日期結果

def human_scroll_and_hover(driver):
    # 模擬隨機滾動頁面（模擬瀏覽）
    scroll_times = random.randint(1, 3)
    for _ in range(scroll_times):
        scroll_px = random.randint(200, 1000)
        driver.execute_script(f"window.scrollBy(0, {scroll_px});")
        time.sleep(random.uniform(0.5, 1.5))

    # 模擬滑鼠移動到隨機元素
    try:
        elements = driver.find_elements(By.CSS_SELECTOR, "h3")
        if elements:
            target = random.choice(elements)
            ActionChains(driver).move_to_element(target).perform()
            time.sleep(random.uniform(0.5, 1.0))
    except:
        pass

def get_result_count_for_date(driver, query, date_str):
    url = (
        f"https://www.google.com/search?q={query}"
        f"&tbs=cdr:1,cd_min:{date_str},cd_max:{date_str}"
        f"&hl=en&lr=lang_en"
    )
    driver.get(url)

    # 模擬人類行為：小滾動與滑鼠 hover
    time.sleep(random.uniform(1, 2))
    human_scroll_and_hover(driver)

    # 嘗試點擊「工具」
    try:
        tools_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "hdtb-tls"))
        )
        ActionChains(driver).move_to_element(tools_btn).pause(0.5).click().perform()
        time.sleep(random.uniform(1.0, 2.0))
    except Exception as e:
        print("⚠️ 無法點擊工具按鈕：", e)

    # 再滾動一點，模擬人類檢查
    human_scroll_and_hover(driver)

    # 嘗試抓 result-stats
    try:
        stats = WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, "result-stats"))
        )
        return stats.text
    except Exception as e:
        print("❌ 找不到 result-stats：", e)
        return "找不到結果數"

def main():
    # 模擬真實瀏覽器設定
    options = uc.ChromeOptions()
    options.add_argument("--lang=en-US")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")
    
    driver = uc.Chrome(headless=False, options=options)

    file_path = "../data/tweets/count/chrome/"
    os.makedirs(file_path, exist_ok=True)
    json_path = f"{file_path}{COIN_SHORT_NAME}_google_search_counts.json"

    current_date = START_DATE
    for _ in range(DAY_COUNT):
        date_str = current_date.strftime("%m/%d/%Y")
        print(f"\n🔍 查詢日期 {date_str}...")
        countStr = get_result_count_for_date(driver, QUERY, date_str)
        print(f"📊 搜尋結果：{countStr}")


        # 擷取搜尋數字，像 "About 1,170 results" → 1170 (支援 comma 和 dot）
        match = re.search(r"About ([\d,\.]+)", countStr)
        if match:
            try:
                number_str = match.group(1).replace(",", "").replace(".", "")
                count = int(float(number_str))
            except ValueError:
                count = 0
        else:
            count = 0


        results.append({
            "date": current_date.strftime("%Y-%m-%d"),
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


        wait_time = random.uniform(7, 15)
        print(f"⏳ 等待 {wait_time:.2f} 秒模擬真人...")
        time.sleep(wait_time)

        current_date += timedelta(days=1)

    driver.quit()
    os._exit(0)

if __name__ == "__main__":
    main()