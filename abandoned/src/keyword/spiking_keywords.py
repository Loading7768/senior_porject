import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# === 1. 讀取每個 result.json 檔 ===
ranking_dict = {}  # key: date, value: {keyword: rank}

json_paths = sorted(glob("../data/keyword/*/result.json"))  # 根據你給的結構

for path in json_paths:
    # 從路徑中擷取日期，例如 ../data/keyword/2025-07-25/result.json
    date = os.path.basename(os.path.dirname(path))
    with open(path, 'r', encoding='utf-8') as f:
        keywords = json.load(f)
        ranking_dict[date] = {}
        for rank, entry in enumerate(keywords, start=1):
            ranking_dict[date][entry['keyword']] = rank

# === 2. 整理為 DataFrame ===
df = pd.DataFrame(ranking_dict).T  # 日期為 index，詞為 columns，值為排名
df = df.sort_index()  # 日期排序

# === 3. 建立儲存資料夾 ===
output_dir = "../outputs/figures/spiking_keywords"
os.makedirs(output_dir, exist_ok=True)

# === 4. 每個關鍵字畫一張圖 ===
for keyword in tqdm(df.columns):
    plt.figure(figsize=(10, 5))
    
    values = df[keyword]
    dates = values.index

    # 1. 折線圖（只連有值的）
    plt.plot(dates, values, label=keyword, linewidth=2, color='blue')

    # 2. 補畫每個點（讓只出現1天也會被看到）
    for i in range(len(values)):
        today_val = values.iloc[i]
        if pd.notna(today_val):
            plt.scatter(dates[i], today_val, color='blue', s=50)

    # 3. 標記首次出現 / 突然消失
    for i in range(len(values)):
        today = values.iloc[i]
        yesterday = values.iloc[i - 1] if i > 0 else None

        # ✅ 首次出現
        if pd.notna(today) and values.iloc[:i].isna().all():
            plt.scatter(dates[i], today, color='green', marker='o', s=100, label='First Appearance')

        # ❌ 突然消失
        elif pd.isna(today) and pd.notna(yesterday):
            plt.scatter(dates[i], yesterday, color='red', marker='x', s=100, label='Sudden Disappearance')

    # === 固定 y 軸為 1～100 ===
    plt.ylim(100.5, 0.5)  # y 軸往上是名次第 1

    # === 基本設定 ===
    plt.gca()
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Rank")
    plt.title(f"Keyword Trend: {keyword}")
    plt.grid(True)
    plt.tight_layout()

    # 避免 legend 重複
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # 儲存圖片
    safe_keyword = "".join(c if c.isalnum() or c in "-_ " else "_" for c in keyword)
    plt.savefig(os.path.join(output_dir, f"{safe_keyword}.png"))
    plt.close()

