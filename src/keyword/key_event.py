from glob import glob
from datetime import datetime
from datetime import timedelta
from dateutil.rrule import rrule, DAILY
import json
import re
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm
import bisect

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
# --------------------parameters--------------------

def load_data():
    '''
    Load:
        - Training keywords
        - Tweets from set time frame
        - Tweets from last week
        - Expansion rate (complete)
    '''
    TRAINING_KEYWORD_PATH = f'../data/keyword/machine_learning/all_keywords.json'

    with open(TRAINING_KEYWORD_PATH, 'r', encoding='utf-8-sig') as file:
        training_keywords = json.load(file)

    TWEET_PATH = f'../data/filtered_tweets/normal_tweets/{COIN_SHORT_NAME}'
    EXPANSION_PATH = f'../data/tweets/count/estimate/{COIN_SHORT_NAME}_estimate.csv'

    expansion_data = pd.read_csv(EXPANSION_PATH, on_bad_lines='skip')

    tweets, expansion_rates, tweet_count_expanded = [], [], 0
    for date in rrule(DAILY, dtstart=START_DATE, until=END_DATE):
        date_str = datetime.strftime(date, '%Y%m%d')
        year, month = date_str[0:4], date_str[4:6]

        formatted_date = datetime.strftime(date, '%Y-%m-%d')
        get_expansion_rate = expansion_data.loc[expansion_data['date'] == formatted_date, 'expansion_ratio']
        expansion_rate = round(get_expansion_rate.iloc[0]) if not get_expansion_rate.empty else 1

        with open(f'{TWEET_PATH}/{year}/{month}/{COIN_SHORT_NAME}_{date_str}_normal.json', 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
            content = data.get(JSON_DICT_NAME, [])
            tweets += content
            expansion_rates += [expansion_rate] * len(content)
            tweet_count_expanded += len(content) * expansion_rate

    tweets_from_last_week = []
    LAST_WEEK_START = START_DATE - timedelta(days=7)
    LAST_WEEK_END = START_DATE - timedelta(days=1)
    for date in rrule(DAILY, dtstart=LAST_WEEK_START, until=LAST_WEEK_END):
        date_str = datetime.strftime(date, '%Y%m%d')
        year, month = date_str[0:4], date_str[4:6]
        with open(f'{TWEET_PATH}/{year}/{month}/{COIN_SHORT_NAME}_{date_str}_normal.json', 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
            tweets_from_last_week += data.get(JSON_DICT_NAME, [])
    
    return training_keywords, tweets, tweets_from_last_week, expansion_rates, tweet_count_expanded

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

def filter_and_compute_df(N, training_keywords, tokenized_tweets, tokens_from_last_week, expansion_rates):
    '''
    Args:
        N: corpus(tweets) size
        tokenized_tweets: a list of tokens from each tweets

    Returns:
        df: dictionary storing a token's df.
    '''
    df = defaultdict(int)

    for tweet_tokens, expansion_rate in tqdm(list(zip(tokenized_tweets, expansion_rates)), desc='Computing df...'):
        for token in tweet_tokens:
            if token in training_keywords and token not in tokens_from_last_week:
                df[token] += expansion_rate

    df = sorted(df.items(), key=lambda x: x[1])
    
    return [i + (i[1] / N,) for i in df]

def save_results(df):
    OUTPUT_PATH = f'../data/keyword/key_event_words.json'

    keywords = [i[0] for i in df]

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as file:
        json.dump(keywords, file)

def main():
    training_keywords, tweets, tweets_from_last_week, expansion_rates, tweet_count_expanded = load_data()
    tokens = tokenize_tweets(tweets)
    tokens_from_last_week = tokenize_tweets(tweets_from_last_week)
    df = filter_and_compute_df(tweet_count_expanded, training_keywords, tokens, tokens_from_last_week, expansion_rates)
    save_results(df)

    for i in df: print(i)

if __name__ == '__main__':
    main()