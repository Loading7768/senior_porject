import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter



'''可修改參數'''
ALL_JSON_DATA = "./data/combined/DOGE_2025_3.json"  # 設定要分群的 json 檔

CLUSTER_FILENAME = "DOGE_2025_3"  # 設定要設為最終檔案的名稱

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱

CLUSTER_NUMBER = 30  # 設定要分成幾群 (cluster)
'''可修改參數'''



# 讀取你上傳的 JSON 檔案
with open(ALL_JSON_DATA, "r", encoding="utf-8-sig") as f:
    data = json.load(f)

tweets = data[JSON_DICT_NAME]
texts = [tweet["text"] for tweet in tweets]

# 使用 BERT 模型
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# KMeans 分群
kmeans = KMeans(n_clusters=CLUSTER_NUMBER, random_state=42)  # random_state: 為了每次執行 K-means 時結果都相同 -> 數值設多少都沒關係
labels = kmeans.fit_predict(embeddings)

# 加入分群結果
for tweet, label in zip(tweets, labels):
    tweet["cluster"] = int(label)

# 根據 cluster 排序
tweets.sort(key=lambda x: x["cluster"])  # lambda: 建立臨時小函式

# 儲存分群結果
newFile = f"./data/clustered/clustered_{CLUSTER_NUMBER}_{CLUSTER_FILENAME}.json"
with open(newFile, "w", encoding="utf-8-sig") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ 分群完成，已儲存為 {newFile}")

# 先收集所有的 cluster 標籤
cluster_labels = [tweet["cluster"] for tweet in tweets]

# 統計每個 cluster 出現的次數
cluster_counts = Counter(cluster_labels)

# 印出每個 cluster 的數量
for cluster_id, count in sorted(cluster_counts.items()):
    print(f"Cluster {cluster_id}: {count} tweets")
