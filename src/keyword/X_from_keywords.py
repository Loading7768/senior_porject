import json
import os
import numpy as np
from pathlib import Path
import sys
from glob import glob
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm
from datetime import datetime
from scipy.sparse import coo_matrix, save_npz
import pickle

# === 匯入 config ===
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import JSON_DICT_NAME, COIN_SHORT_NAME


'''可修改參數'''
IS_FILTERED = False  # 看是否有分 normal 與 bot

IS_RUN_AUGUST = False  # 看現在是不是要跑 2025/08 的資料  START_DATE, END_DATE 會固定

# === 參數設定 ===
DATA_DIR = "../data/keyword/machine_learning"
OUT_DIR = "../data/ml/dataset/keyword"

# === 自訂時間範圍 (格式：YYYY/MM) ===
START_DATE = "2013/12"   # 自訂開始年月
END_DATE   = "2025/07"   # 結束年月

'''可修改參數'''

os.makedirs(OUT_DIR, exist_ok=True)

if IS_FILTERED:
    TWEET_DIR = f"../data/filtered_tweets/normal_tweets/{COIN_SHORT_NAME}/*"
else
    TWEET_DIR = f"../data/tweets/{COIN_SHORT_NAME}/*"

if IS_RUN_AUGUST:
    START_DATE = "2025/08"
    END_DATE   = "2025/08"



# === 自訂 tokenize 函式 ===
def tokenize_tweets(tweets):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    STOPWORDS = set(stopwords.words('english'))
    tokenized_tweets = []

    for tweet in tweets:
        tweet_text = tweet.get('text', "")
        tokens = tokenizer.tokenize(tweet_text)
        unique_tokens = set(token for token in tokens if token not in STOPWORDS and token.isalnum())
        tokenized_tweets.append(unique_tokens)

    return tokenized_tweets


# === 讀取單一幣種的詞彙表 ===
if IS_FILTERED:
    json_path = os.path.join(DATA_DIR, "all_keywords.json")
else:
    json_path = os.path.join(DATA_DIR, "all_keywords_non_filtered.json")

with open(json_path, "r", encoding="utf-8-sig") as f:
    vocab = json.load(f)

all_vocab = list(vocab)
print(f"詞彙數量: {len(all_vocab)}")

word2idx = {w: i for i, w in enumerate(all_vocab)}

# === 找出所有 JSON 檔案 ===
start_year, start_month = map(int, START_DATE.split('/'))
end_year, end_month = map(int, END_DATE.split('/'))

all_files = []
for year_folder in glob(TWEET_DIR):
    year = int(os.path.basename(year_folder))
    if year < start_year or year > end_year:
        continue

    for month_folder in glob(os.path.join(year_folder, "*")):
        month = int(os.path.basename(month_folder))
        if (year == start_year and month < start_month) or (year == end_year and month > end_month):
            continue

        # 找出這個月的所有 JSON
        if IS_FILTERED:
            pattern = os.path.join(month_folder, f"{COIN_SHORT_NAME}_*_normal.json")
        else:
            pattern = os.path.join(month_folder, f"{COIN_SHORT_NAME}_*.json")
        all_files.extend(glob(pattern))

json_files = all_files
print(f"找到 {len(json_files)} 個檔案可處理")

# === 用稀疏矩陣的三元組格式 (row, col, data) ===
rows, cols, data_vals = [], [], []
ids = []

row_idx = 0
for jf in tqdm(json_files, desc="處理推文檔案"):
    with open(jf, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    tweets = data.get(JSON_DICT_NAME, [])
    tokenized_tweets = tokenize_tweets(tweets)

    for tw, tokens in zip(tweets, tokenized_tweets):
        raw_date = tw.get("created_at", "")
        try:
            dt = datetime.strptime(raw_date, "%a %b %d %H:%M:%S %z %Y")
            date = dt.strftime("%Y-%m-%d")
        except Exception:
            date = ""

        number = tw.get("tweet_count", '')
        
        for tok in tokens:
            if tok in word2idx:
                rows.append(row_idx)
                cols.append(word2idx[tok])
                data_vals.append(1)   # 出現一次就 +1 (因為 tokenized 已去重)

        ids.append((COIN_SHORT_NAME, date, number))
        row_idx += 1

# === 建立 CSR 稀疏矩陣 ===
X_sparse = coo_matrix((data_vals, (rows, cols)), shape=(row_idx, len(all_vocab)), dtype=np.int32)
X_sparse = X_sparse.tocsr()

# ============================================================
# === 存檔 (矩陣 + 日期 + vocab) ===
# ============================================================
if IS_FILTERED:
    if not IS_RUN_AUGUST:
        X_sparse_output = f"{COIN_SHORT_NAME}_X_sparse.npz"
        ids_output = f'{COIN_SHORT_NAME}_ids.pkl'
    else:
        X_sparse_output = f"{COIN_SHORT_NAME}_X_sparse_202508.npz"
        ids_output = f'{COIN_SHORT_NAME}_ids_202508.pkl'
else:
    if not IS_RUN_AUGUST:
        X_sparse_output = f"{COIN_SHORT_NAME}_X_sparse_non_filtered.npz"
        ids_output = f'{COIN_SHORT_NAME}_ids_non_filtered.pkl'
    else:
        X_sparse_output = f"{COIN_SHORT_NAME}_X_sparse_non_filtered_202508.npz"
        ids_output = f'{COIN_SHORT_NAME}_ids_non_filtered_202508.pkl'

save_npz(os.path.join(OUT_DIR, f"{X_sparse_output}"), X_sparse)
with open(os.path.join(OUT_DIR, f'{ids_output}'), 'wb') as file:
    pickle.dump(ids, file)

print(f"原始矩陣: X.shape = {X_sparse.shape}, 資料量 = {len(ids)}")


"""
#存成 txt
X_sparse = load_npz("../data/ml/dataset/keywords/PEPE_X_sparse.npz")
dates = np.load("../data/ml/dataset/keywords/PEPE_dates.npy")

# 取出 COO 格式
X_coo = X_sparse.tocoo()

# 存三元組格式 (row, col, val)
np.savetxt("../data/ml/dataset/keywords/PEPE_X_triplets.txt",
           np.vstack((X_coo.row, X_coo.col, X_coo.data)).T,
           fmt="%d")

# 存日期
np.savetxt("../data/ml/dataset/keywords/PEPE_dates.txt", dates, fmt="%s")

print("✅ 已經輸出成 txt 檔")
"""