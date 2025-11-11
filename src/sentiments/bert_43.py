# for price classifier, sentiments by day using voting by most common and most confident(highest socore average)
from transformers import pipeline
import numpy as np
from collections import defaultdict
from glob import glob
from tqdm import tqdm
import re
from datetime import datetime
import json
from pathlib import Path
import os


# ---------------------------------------------------------------------------
# Load the pre-trained model and tokenizer
model = 'borisn70/bert-43-multilabel-emotion-detection'
tokenizer = 'borisn70/bert-43-multilabel-emotion-detection'

# Create a pipeline for sentiment analysis
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# ---------------------------------------------------------------------------


def load_tweets(coin_short_name, json_dict_name):
    TWEET_PATH = f'../data/filtered_tweets/normal_tweets/{coin_short_name}/*/*/*.json'

    tweet_files = glob(TWEET_PATH)
    tweet_files = sorted(tweet_files)

    tweet_by_date = defaultdict(list)
    for tf in tqdm(tweet_files, desc='Loading tweets...'):
        match_date = re.search(r'\d{8}', tf)
        date = datetime.strptime(match_date.group(), '%Y%m%d')
        if date > datetime(2025, 7, 31):
            continue

        try:
            with open(tf, 'r', encoding="utf-8-sig") as file:
                data = json.load(file)
                content = data.get(json_dict_name, [])
                texts = [entry['text'] for entry in content if 'text' in entry]
                tweet_by_date[match_date] += texts

        except(json.JSONDecodeError, FileNotFoundError) as e:
            print(f'Error loading {tf}: {e}')

    return tweet_by_date


def process_sentiments(tweet_by_date):
    voted_results = []

    for date, tweets in tqdm(tweet_by_date.items(), desc='processing...'):
        if not tweets: continue
        tweet_count = len(tweets)
        sents = {
            "admiration": 0.0, "amusement": 0.0, "anger": 0.0, "annoyance": 0.0, "approval": 0.0,
            "caring": 0.0, "confusion": 0.0, "curiosity": 0.0, "desire": 0.0, "disappointment": 0.0,
            "disapproval": 0.0, "disgust": 0.0, "embarrassment": 0.0, "excitement": 0.0, "fear": 0.0,
            "gratitude": 0.0, "grief": 0.0, "joy": 0.0, "love": 0.0, "nervousness": 0.0,
            "optimism": 0.0, "pride": 0.0, "realization": 0.0, "relief": 0.0, "remorse": 0.0,
            "sadness": 0.0, "surprise": 0.0, "neutral": 0.0, "worry": 0.0, "happiness": 0.0,
            "fun": 0.0, "hate": 0.0, "autonomy": 0.0, "safety": 0.0, "understanding": 0.0,
            "empty": 0.0, "enthusiasm": 0.0, "recreation": 0.0, "sense of belonging": 0.0,
            "meaning": 0.0, "sustenance": 0.0, "creativity": 0.0, "boredom": 0.0
        }

        results = nlp(tweets)
        for r in results:
            sents[r['label']] += 1

        for key, value in sents.items():
            if value > 0.0:
                sents[key] /= tweet_count

        voted_results.append(list(sents.values()))

    return voted_results


def save_results(coin_short_name, results):
    OUTPUT_PATH = Path(f'../data/sentiments/{coin_short_name}_bert_43.npy')
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    np.save(OUTPUT_PATH, results)


def main():
    COIN_SHORT_NAMES = ['TRUMP', 'PEPE', 'DOGE']
    JSON_DICT_NAMES = [
        '(officialtrump OR "official trump" OR "trump meme coin" OR "trump coin" OR trumpcoin OR $TRUMP OR "dollar trump")',
        'PEPE', 
        'dogecoin']
    for csm, jdn in zip(COIN_SHORT_NAMES, JSON_DICT_NAMES):
        print(csm)
        tweet_by_date = load_tweets(csm, jdn)
        voted_results = process_sentiments(tweet_by_date)
        save_results(csm, voted_results)


if __name__ == '__main__':
    main()