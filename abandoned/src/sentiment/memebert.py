from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os, json, codecs, csv
from datetime import datetime
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

# === 設定變數 ===
YEAR = "2025"
MONTH = "01"
INPUT_DIR = f"../data/filtered_tweets/normal_tweets/{YEAR}/{MONTH}"
POS_DIR = f"../data/sentiment/{COIN_SHORT_NAME}/{YEAR}/{MONTH}/positive"
NEU_DIR = f"../data/sentiment/{COIN_SHORT_NAME}/{YEAR}/{MONTH}/neutral"
NEG_DIR = f"../data/sentiment/{COIN_SHORT_NAME}/{YEAR}/{MONTH}/negative"
CSV_OUTPUT_DIR = f"../data/sentiment/{COIN_SHORT_NAME}/summary"
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEU_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# === 載入 BERT 模型（MemeBERT 或類似模型）===
model_name = "cardiffnlp/twitter-roberta-base-sentiment"  # 推薦用於推文情緒分析
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    label_id = torch.argmax(probs, dim=1).item()
    return ["negative", "neutral", "positive"][label_id]

# === 開始處理 ===
summary_data = []
for file in sorted(os.listdir(INPUT_DIR)):
    if not file.endswith(".json"):
        continue
    date_str = file.replace(f"{COIN_SHORT_NAME}_", "").replace("_normal.json", "")
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        continue
    filepath = os.path.join(INPUT_DIR, file)
    try:
        with codecs.open(filepath, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)
        tweets = raw.get(JSON_DICT_NAME, [])
    except:
        continue

    pos, neu, neg = [], [], []
    for tweet in tweets:
        text = tweet.get("text", "")
        if not text.strip():
            continue
        sentiment = classify_sentiment(text)
        if sentiment == "positive":
            pos.append(tweet)
        elif sentiment == "negative":
            neg.append(tweet)
        else:
            neu.append(tweet)

    def save_json(subfolder, name, content):
        path = os.path.join(subfolder, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    save_json(POS_DIR, f"{COIN_SHORT_NAME}_{date_str}_positive", pos)
    save_json(NEU_DIR, f"{COIN_SHORT_NAME}_{date_str}_neutral", neu)
    save_json(NEG_DIR, f"{COIN_SHORT_NAME}_{date_str}_negative", neg)

    summary_data.append({
        "date": date_str,
        "total_tweets": len(pos) + len(neu) + len(neg),
        "positive": len(pos),
        "neutral": len(neu),
        "negative": len(neg)
    })

# === 寫出 CSV ===
csv_filename = f"{COIN_SHORT_NAME}_{YEAR}_{MONTH}_sentiment_summary.csv"
csv_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["date", "total_tweets", "positive", "neutral", "negative"])
    writer.writeheader()
    for row in summary_data:
        writer.writerow(row)
