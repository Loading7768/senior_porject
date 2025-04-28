import json



'''可修改參數'''
CLUSTER_FILENAME = "d:/senior_project/Kmeans/data/clustered/clustered_30_DOGE_2025_2.json"  # 設定要排序的檔案名稱

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱
'''可修改參數'''



# 讀取你上傳的 JSON 檔案
with open(CLUSTER_FILENAME, "r", encoding="utf-8-sig") as f:
    data = json.load(f)

tweets = data[JSON_DICT_NAME]

# 根據 cluster 排序
tweets.sort(key=lambda x: x["cluster"])  # lambda: 建立臨時小函式

# 儲存分群結果
with open(CLUSTER_FILENAME, "w", encoding="utf-8-sig") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ 已重新排序，已儲存為 {CLUSTER_FILENAME}")