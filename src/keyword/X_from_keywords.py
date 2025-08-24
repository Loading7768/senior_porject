import json
import os
import numpy as np
from pathlib import Path
import sys
import glob
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from tqdm import tqdm
from datetime import datetime
from scipy.sparse import coo_matrix, save_npz, load_npz

# === 匯入 config ===
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import JSON_DICT_NAME, COIN_SHORT_NAME

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


# === 參數設定 ===
DATA_DIR = "../data/keywords/machine_learning"
TWEET_DIR = f"../data/filtered_tweets/normal_tweets/*"
OUT_DIR = "../data/ml/dataset/keywords"
os.makedirs(OUT_DIR, exist_ok=True)

# === 讀取單一幣種的詞彙表 ===
json_path = os.path.join(DATA_DIR, "all_keywords.json")
with open(json_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

all_vocab = sorted(list(vocab))
print(f"詞彙數量: {len(all_vocab)}")

word2idx = {w: i for i, w in enumerate(all_vocab)}

# === 找出所有 JSON 檔案 ===
json_files = glob.glob(os.path.join(TWEET_DIR, "*", f"{COIN_SHORT_NAME}_*_normal.json"))
print(f"找到 {len(json_files)} 個檔案可處理")

# === 用稀疏矩陣的三元組格式 (row, col, data) ===
rows, cols, data_vals = [], [], []
dates = []

row_idx = 0
for jf in tqdm(json_files, desc="處理推文檔案"):
    with open(jf, "r", encoding="utf-8") as f:
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

        for tok in tokens:
            if tok in word2idx:
                rows.append(row_idx)
                cols.append(word2idx[tok])
                data_vals.append(1)   # 出現一次就 +1 (因為 tokenized 已去重)

        dates.append(date)
        row_idx += 1

# === 建立 CSR 稀疏矩陣 ===
X_sparse = coo_matrix((data_vals, (rows, cols)), shape=(row_idx, len(all_vocab)), dtype=np.int32)
X_sparse = X_sparse.tocsr()

print(f"原始矩陣: X.shape = {X_sparse.shape}, 日期數 = {len(dates)}")

# ============================================================
# === 功能 1: 刪掉只出現 <= min_count 次的詞彙 (刪 column) ===
# ============================================================
min_count = 2   # 可以改成 1, 2, 3
col_sums = np.array(X_sparse.sum(axis=0)).ravel()
valid_cols = np.where(col_sums > min_count)[0]

X_sparse = X_sparse[:, valid_cols]
filtered_vocab = [all_vocab[i] for i in valid_cols]

print(f"過濾後詞彙數量: {len(filtered_vocab)} (原本 {len(all_vocab)})")

# === 保留一份原始日期，不要被覆蓋 ===
dates = np.array(dates)       # 全部推文的日期
dates_all = dates.copy()      # 存一份完整的原始日期

# ============================================================
# === 功能 2: 刪掉沒有任何關鍵詞的推文 (刪 row) ===
# ============================================================
row_sums = np.array(X_sparse.sum(axis=1)).ravel()
valid_rows = np.where(row_sums > 0)[0]
invalid_rows = np.where(row_sums == 0)[0]

# 保留有效 row
X_sparse = X_sparse[valid_rows, :]
dates_kept = dates_all[valid_rows]
dates_removed = dates_all[invalid_rows]

print(f"刪掉 {len(invalid_rows)} 筆沒有關鍵詞的推文，保留 {len(valid_rows)} 筆")

# === 輸出被刪掉的推文資訊 (原始 index + 日期) ===
removed_path = os.path.join(OUT_DIR, f"{COIN_SHORT_NAME}_removed_tweets.txt")
with open(removed_path, "w", encoding="utf-8") as f:
    for idx in invalid_rows:
        f.write(f"{idx}\t{dates_all[idx]}\n")
print(f"✅ 已輸出被刪掉的推文資訊到 {removed_path}")

# ============================================================
# === 存檔 (矩陣 + 日期 + vocab) ===
# ============================================================
save_npz(os.path.join(OUT_DIR, f"{COIN_SHORT_NAME}_X_sparse_filtered.npz"), X_sparse)
np.save(os.path.join(OUT_DIR, f"{COIN_SHORT_NAME}_dates_filtered.npy"), dates_kept)

with open(os.path.join(OUT_DIR, f"{COIN_SHORT_NAME}_filtered_vocab.json"), "w", encoding="utf-8") as f:
    json.dump(filtered_vocab, f, ensure_ascii=False, indent=2)

print(f"完成過濾後: X.shape = {X_sparse.shape}, 日期數 = {len(dates_kept)}")


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