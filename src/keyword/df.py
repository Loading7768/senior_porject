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
from tqdm import tqdm

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

def load_tweets():
    TWEET_PATH = f'../data/filtered_tweets/normal_tweets/{COIN_SHORT_NAME}/*/*/*'
    tweet_files = glob(TWEET_PATH)

    tweets = []
    for tf in tqdm(tweet_files, desc='Loading tweets...'):
        try:
            with open(tf, 'r', encoding="utf-8") as file:
                data = json.load(file)
                tweets = tweets + data.get(JSON_DICT_NAME, [])

        except(json.JSONDecodeError, FileNotFoundError) as e:
            print(f'Error loading {tf}: {e}')

    return tweets

def tokenize_tweets(tweets):
    '''
    Break every tweets into tokens via tokenizer.
    '''
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    STOPWORDS = set(stopwords.words('english'))
    tokenized_tweets = []

    for tweet in tqdm(tweets, desc='Tokeninzing...'):
        tweet_text = tweet.get('text')
        tokens = tokenizer.tokenize(tweet_text)
        unique_tokens = set(token for token in tokens if token not in STOPWORDS and token.isalpha())
        tokenized_tweets.append(unique_tokens)

    return tokenized_tweets

def compute_df(N, tokenized_tweets):
    '''
    Args:
        N: corpus(tweets) size
        tokenized_tweets: a list of tokens from each tweets

    Returns:
        idf: dictionary storing a token's idf.
        df: dictionary storing a token's df.
    '''
    df = defaultdict(int)

    for tweet_tokens in tqdm(tokenized_tweets, desc='Computing df...'):
        for token in tweet_tokens:
            df[token] += 1

    return sorted(df.items(), key=lambda x: x[1], reverse=True)

def main():
    tweets = load_tweets()
    tokens = tokenize_tweets(tweets)
    df = compute_df(len(tweets), tokens)

    for i in range(100):
        print(df[i])

if __name__ == '__main__':
    main()