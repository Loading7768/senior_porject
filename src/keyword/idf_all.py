'''
This script finds global keyword from the corpus(tweets) by IDF(Inverse Document Frequency) and casting votes.

This implementation ignores the TF part since all tweets are quite short,
words are unlikley to show up more than once,
redering the value negligible in the same tweet.
'''

from glob import glob
from datetime import datetime
from dateutil.rrule import rrule, DAILY
import bisect
import json
import re
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import math
import os
import numpy as np
import pandas as pd

'''
Importing parameters from project_root/config.py
'''
from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import JSON_DICT_NAME, COIN_SHORT_NAME

# --------------------parameters--------------------
START_DATE = datetime(2025, 1, 18)
END_DATE = datetime(2025, 7, 31)

KEYWORD_PER_DAY = 100
# --------------------parameters--------------------

def load_tweets(date):
    '''
    Load tweets from a entire given date.
    '''
    TWEET_PATH = Path('../data/filtered_tweets/normal_tweets')
    date_str = datetime.strftime(date, '%Y%m%d')
    year, month = date_str[0:4], date_str[4:6]
    json_file = Path(f'{TWEET_PATH}/{year}/{month}/{COIN_SHORT_NAME}_{date_str}_normal.json')

    tweets = []
    try:
        with open(json_file, 'r', encoding="utf-8") as file:
            data = json.load(file)
            data = data.get(JSON_DICT_NAME, [])

            for tweet in data:
                tweets.append(tweet)
    except(json.JSONDecodeError, FileNotFoundError) as e:
        print(f'Error loading {json_file}: {e}')

    return tweets

def tokenize_tweets(tweets):
    '''
    Break every tweets into tokens via tokenizer.
    '''
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    STOPWORDS = set(stopwords.words('english'))
    tokenized_tweets = []

    for tweet in tweets:
        tweet_text = tweet.get('text')
        tokens = tokenizer.tokenize(tweet_text)
        unique_tokens = set(token for token in tokens if token not in STOPWORDS and token.isalnum())
        tokenized_tweets.append(unique_tokens)

    return tokenized_tweets

def compute_idf(N, tokenized_tweets):
    '''
    Args:
        N: corpus(tweets) size
        tokenized_tweets: a list of tokens from each tweets

    Returns:
        idf: dictionary storing a token's idf.
        df: dictionary storing a token's df.
    '''
    df = defaultdict(int)

    for tweet_tokens in tokenized_tweets:
        for token in tweet_tokens:
            df[token] += 1

    idf = {token: math.log((N) / (df_count+1)) + 1 for token, df_count in df.items()}

    return idf

def find_top_keywords(tokenized_tweets, idf, top_n=3):
    '''
    find gloabl keyword by casting votes with top n idf from each tweets as candidates.
    Args:
        top_n (int): determines the number of candidates pulled from each tweets.
    '''
    daily_global_keywords = defaultdict(int)

    for tweet_tokens in tokenized_tweets:
        token_idfs = [(token, idf.get(token, 0)) for token in tweet_tokens]
        top_keywords = sorted(token_idfs, key=lambda x: x[1], reverse=True)[:top_n]
        for k in top_keywords:
            daily_global_keywords[k[0]] += 1

    daily_global_keywords = sorted(daily_global_keywords.items(), key=lambda x: x[1], reverse=True)

    return [k[0] for k in daily_global_keywords[:100]]
def save_result(global_keywords):
    '''
    Write a json file storing the results.
    '''
    OUTPUT_PATH = Path(f'../data/keyword/machine_learning/{COIN_SHORT_NAME}_keywords.json')
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as file:
        json.dump(global_keywords, file)

def main():
    global_keywords = []

    for date in rrule(DAILY, dtstart=START_DATE, until=END_DATE):
        print(f'Processing {datetime.strftime(date, '%Y%m%d')}...')
        tweets = load_tweets(date)
        if tweets is None:
            print(f'No data in {datetime.strftime(date, '%Y%m%d')}, skipping.')
            continue
        print(f'Successfully loaded {len(tweets)} tweets.')
        print('Tokeninzing...')
        tokens = tokenize_tweets(tweets)
        print('Calculating idf...')
        idf = compute_idf(len(tweets), tokens)
        print('Gathering keywords...')
        daily_top_keywords = find_top_keywords(tokens, idf)
        global_keywords = global_keywords + daily_top_keywords
        print('-' * 100)

    print('Writing json...')
    global_keywords = list(set(global_keywords))
    save_result(global_keywords)
    print(f'🖒🖒🖒 Done. Collected {len(global_keywords)} keywords.')

if __name__ == '__main__':
    main()
