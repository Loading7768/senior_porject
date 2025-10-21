from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt



'''可修改參數'''
ALL_JSON_DATA = "./data/combined/DOGE_2025_2.json"  # 設定要把所有指定推文儲存到的地方

CLUSTER_NUMBER = 100  # 判斷從 k = 2 ~ CLUSTER_NUMBER 中，分幾群的評分最高

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱
'''可修改參數'''



with open(ALL_JSON_DATA, 'r', encoding='utf-8-sig') as file:
    data = json.load(file)

# 讀取 json 檔中的 "text"
texts = [t["text"] for t in data[JSON_DICT_NAME]]

# 使用 BERT 嵌入
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

scores = []  # 放所有評分

# 測試不同群數
for k in tqdm(range(2, CLUSTER_NUMBER + 1), desc="Evaluating KMeans"):  # tqdm: 用來產生進度條
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    silhouette = silhouette_score(embeddings, labels)
    # calinski = calinski_harabasz_score(embeddings, labels)
    davies = davies_bouldin_score(embeddings, labels)

    scores.append({
        "k": k,
        "silhouette": silhouette,
        # "calinski": calinski,
        "davies": davies
    })
    # , calinski = {calinski:.4f}
    print(f"\n{k} 群：silhouette = {silhouette:.4f}, davies = {davies:.4f}")

# 印出結果
df = pd.DataFrame(scores)
df["davies_inv"] = 1 / df["davies"]  # 轉換為越高越好 -> 原本 davies 是數值越低越好

df["score_combined"] = (  # 簡單加權平均分數（你也可以自訂）
    df["silhouette"].rank(ascending=False) +  # ascending=False: 從數值高到低排名 排名(1, 2, 3,...)
    # df["calinski"].rank(ascending=False) +
    df["davies_inv"].rank(ascending=False)
)

best_k_row = df.loc[df["score_combined"].idxmin()]
print("✅ 建議最佳群數：", int(best_k_row["k"]))


# 畫出分數圖，輔助人工判斷
plt.figure(figsize=(12,6))
plt.plot(df["k"], df["silhouette"], label="Silhouette")
# plt.plot(df["k"], df["calinski"]/df["calinski"].max(), label="Calinski (scaled)")
plt.plot(df["k"], df["davies_inv"]/df["davies_inv"].max(), label="1/Davies (scaled)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Score (normalized)")
plt.title("Clustering Score Trends")
plt.legend()
plt.grid(True)
plt.show()
