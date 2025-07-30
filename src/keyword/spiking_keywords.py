import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

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
# df = df.fillna(None)

# === 3. 畫圖（所有關鍵字） ===
plt.figure(figsize=(15, 8))

for keyword in df.columns:
    plt.plot(df.index, df[keyword], label=keyword)

plt.gca().invert_yaxis()  # 排名越高在越上面
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Rank")
plt.title("Keyword Ranking Trend Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()
