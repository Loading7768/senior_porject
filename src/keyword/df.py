from glob import glob
from datetime import datetime
import json
import re
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm
import bisect

# --------------------parameters--------------------
'''
Whether to only use 'normal tweets' or not.
'''
IS_FILTERED = False
# --------------------parameters--------------------

def load_data(coin_short_name, json_dict_name, total_expansion_rate):
    TWEET_PATH = f'../data/filtered_tweets/normal_tweets/{coin_short_name}/*/*/*.json' if IS_FILTERED else f'../data/tweets/{coin_short_name}/*/*/*.json'
    EXPANSION_PATH = f'../data/tweets/count/estimate/{coin_short_name}_estimate.csv'

    tweet_files = glob(TWEET_PATH)
    expansion_data = pd.read_csv(EXPANSION_PATH, on_bad_lines='skip')

    tweets, expansion_rates, tweet_count_expanded = [], [], 0

    for tf in tqdm(tweet_files, desc='Loading tweets...'):
        match_date = re.search(r'\d{8}', tf)
        date = datetime.strptime(match_date.group(), '%Y%m%d')
        if date > datetime(2025, 7, 31):
            continue

        formatted_date = datetime.strftime(date, '%Y-%m-%d')
        get_expansion_rate = expansion_data.loc[expansion_data['date'] == formatted_date, 'expansion_ratio']
        expansion_rate = get_expansion_rate.iloc[0] if not get_expansion_rate.empty else 1

        try:
            with open(tf, 'r', encoding="utf-8-sig") as file:
                data = json.load(file)
                content = data.get(json_dict_name, [])
                tweets += content
                expansion_rates += [expansion_rate * total_expansion_rate] * len(content)
                tweet_count_expanded += len(content) * expansion_rate * total_expansion_rate

        except(json.JSONDecodeError, FileNotFoundError) as e:
            print(f'Error loading {tf}: {e}')

    return tweets, expansion_rates, tweet_count_expanded

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

def compute_df(N, tokenized_tweets, expansion_rates):
    '''
    Args:
        N: corpus(tweets) size
        tokenized_tweets: a list of tokens from each tweets

    Returns:
        idf: dictionary storing a token's idf.
        df: dictionary storing a token's df.
    '''
    df = defaultdict(float)

    for tweet_tokens, expansion_rate in tqdm(zip(tokenized_tweets, expansion_rates), desc='Computing df...'):
        for token in tweet_tokens:
            df[token] += expansion_rate

    df = sorted(df.items(), key=lambda x: x[1])
    
    return [i + (i[1] / N,) for i in df]

def filter_and_save(df):
    UPPER_BOUND = 0.05
    LOWER_BOUND = 0.001

    upper = bisect.bisect_right([i[2] for i in df], UPPER_BOUND)
    lower = bisect.bisect_left([i[2] for i in df], LOWER_BOUND)

    OUTPUT_PATH = f'../data/keyword/machine_learning/all_keywords.json' if IS_FILTERED else f'../data/keyword/machine_learning/all_keywords_non_filtered.json'

    keywords = [i[0] for i in df[lower:upper]]

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as file:
        json.dump(keywords, file)

def main():
    COIN_SHORT_NAMES = ['TRUMP', 'PEPE', 'DOGE']
    JSON_DICT_NAMES = [
        '(officialtrump OR "official trump" OR "trump meme coin" OR "trump coin" OR trumpcoin OR $TRUMP OR "dollar trump")',
        'PEPE', 
        'dogecoin']
    TOTAL_TWEET_EXPANSION_RATES = [11.6778852993119624, 3.10285237528924781, 1] if IS_FILTERED else [10.672908648, 3.2318801, 1]

    tweets, expansion_rates, tweet_count_expanded = [], [], 0
    for csn, jdn, tter in zip(COIN_SHORT_NAMES, JSON_DICT_NAMES, TOTAL_TWEET_EXPANSION_RATES):
        t, er, tce = load_data(csn, jdn, tter)
        tweets += t
        expansion_rates += er
        tweet_count_expanded += tce

    print(len(tweets), tweet_count_expanded)
    tokens = tokenize_tweets(tweets)
    df = compute_df(tweet_count_expanded, tokens, expansion_rates)
    filter_and_save(df)

    for i in df:
        print(i)

if __name__ == '__main__':
    main()