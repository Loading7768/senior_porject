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
        sents = defaultdict(lambda: {'votes': 0, 'score': 0.0})
        results = nlp(tweets)
        for r in results:
            sents[r['label']]['votes'] += 1
            sents[r['label']]['score'] += r['score']

        for label, val in sents.items():
            if val['votes'] > 0: val['score'] /= val['votes']
        sorted_sents = sorted(
            sents.items(),
            key=lambda item: (item[1]['votes'], item[1]['score']),
            reverse=True
        )

        voted_results.append(sorted_sents[0][0])

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
    for csm, jdn in zip(COIN_SHORT_NAMES[1:2], JSON_DICT_NAMES[1:2]):
        print(csm)
        tweet_by_date = load_tweets(csm, jdn)
        voted_results = process_sentiments(tweet_by_date)
        save_results(csm, voted_results)


if __name__ == '__main__':
    main()