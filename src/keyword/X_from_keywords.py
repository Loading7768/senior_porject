#!/usr/bin/env python3
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

# === 匯入 config ===
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import JSON_DICT_NAME, COIN_SHORT_NAME

# === 自訂 tokenize 函式 ===
def tokenize_tweets(tweets):
    '''
    Break every tweets into tokens via tokenizer.
    '''
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
DATA_DIR = "../data/keyword/machine_learning"   # 放 json 的資料夾 (keywords)
TWEET_DIR = f"../data/filtered_tweets/normal_tweets/*" # 推文 JSON 資料夾
OUT_DIR = "../data/keyword/machine_learning"    # 輸出資料夾
os.makedirs(OUT_DIR, exist_ok=True)

# === 讀取單一幣種的詞彙表 ===
json_path = os.path.join(DATA_DIR, "all_keywords.json")
with open(json_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

all_vocab = sorted(list(vocab))
print(f"{COIN_SHORT_NAME} 詞彙數量: {len(all_vocab)}")

# 建立詞彙到 index 的映射
word2idx = {w: i for i, w in enumerate(all_vocab)}

# === 找出所有 JSON 檔案 ===
json_files = glob.glob(os.path.join(TWEET_DIR, "*", f"{COIN_SHORT_NAME}_*_normal.json"))
print(f"找到 {len(json_files)} 個檔案可處理")

X = []
dates = []

# === 逐檔處理，套 tqdm ===
for jf in tqdm(json_files, desc="處理推文檔案"):
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 取出對應幣別的推文陣列
    tweets = data.get(JSON_DICT_NAME, [])

    # 使用新的 tokenize 函式
    tokenized_tweets = tokenize_tweets(tweets)

    for tw, tokens in zip(tweets, tokenized_tweets):
        raw_date = tw.get("created_at", "")
        try:
            dt = datetime.strptime(raw_date, "%a %b %d %H:%M:%S %z %Y")
            date = dt.strftime("%Y-%m-%d")   # 例：2025-07-01
        except Exception:
            date = ""   # 防呆，避免格式錯誤

        vec = np.zeros(len(all_vocab), dtype=int)
        for tok in tokens:
            if tok in word2idx:
                vec[word2idx[tok]] += 1

        X.append(vec)
        dates.append(date)

# === 轉 numpy array & 存檔 ===
X = np.array(X)
dates = np.array(dates)

np.save(os.path.join(OUT_DIR, f"{COIN_SHORT_NAME}_X.npy"), X)
np.save(os.path.join(OUT_DIR, f"{COIN_SHORT_NAME}_dates.npy"), dates)

print(f"完成全部檔案: X.shape = {X.shape}, 日期數 = {len(dates)}")

"""
存成 txt
X = np.load("../data/keyword/machine_learning/PEPE_X.npy")
dates = np.load("../data/keyword/machine_learning/PEPE_dates.npy")

np.savetxt("../data/keyword/machine_learning/PEPE_X.txt", X, fmt="%d")
np.savetxt("../data/keyword/machine_learning/PEPE_dates.txt", dates, fmt="%s")

print("✅ 已經輸出成 txt 檔")
"""