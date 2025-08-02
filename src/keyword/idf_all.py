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

# ----------parameters----------
'''
Top nth global keywords recorded in the result.
'''
GLOBAL_KEYWORDS_AMOUNT = 100

'''
You can modify SET_RANGE to set whether date range is applied,
'False' will use every file in the folder,
'True' will use the data within START_DATE ~ END_DATE(inclusive).

Set the date using datetime(Y, D, M), no need to fill in extra 0s.
'''
SET_RANGE = True
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 1, 1)

'''
DAILY_MODE rums IDF day by day with the specified time frame.
*NOTE* You do need to put SET_RANGE to True. 

Set the date using datetime(Y, D, M), no need to fill in extra 0s.
'''
DAILY_MODE = True
DAILY_START_DATE = datetime(2025, 1, 1)
DAILY_END_DATE = datetime(2025, 6, 30)
# ----------parameters----------

# ----------globla variable----------
'''
Output vector for machine learning. Each row is a different day of top gloabl keywords.
Disgustingly implemented, I know.
This script is a fucking mess and demands a total rewrite.
'''
df_daily = []
feature_names = []
feature_vector = []
# ----------globla variable----------

# load tweets
def load_tweets():
    tweets = []

    # glob every file under ../data/filtered_tweets/normal_tweets/YYYY/mm
    json_files = sorted(glob(f'../data/filtered_tweets/normal_tweets/*/*/*'))
    if not json_files:
        print(f'No files found in the directory.')
        return []
    
    # When in SET_RANGE mode, use binary search to find start and end
    if SET_RANGE:
        file_dates = []
        for json_file in json_files:
            match_date = re.search(r'\d{8}', json_file)
            file_dates.append(datetime.strptime(match_date.group(), '%Y%m%d'))

        left = bisect.bisect_left(file_dates, START_DATE)
        right = bisect.bisect_right(file_dates, END_DATE)
        if left == right:   # no file found with in set range
            print('No files found within the time frame.')
            return []
        json_files = json_files[left:right]

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding="utf-8") as file:
                data = json.load(file)
                data = data.get(JSON_DICT_NAME, [])

                for tweet in data:
                    tweets.append(tweet)
        except(json.JSONDecodeError, FileNotFoundError) as e:
            print(f'Error loading {json_file}: {e}')
            continue

    return tweets

def tokenize_tweets(tweets):
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

    return idf, df

def find_global_keyword(tokenized_tweets, idf, top_n=3):
    '''
    find gloabl keyword by casting votes with top n idf from each tweets as candidates.
    Args:
        top_n (int): determines the number of candidates pulled from each tweets.
    '''
    global_keywords = defaultdict(int)

    for tweet_tokens in tokenized_tweets:
        token_idfs = [(token, idf.get(token, 0)) for token in tweet_tokens]
        top_keywords = sorted(token_idfs, key=lambda x: x[1], reverse=True)[:top_n]
        
        # print(top_keywords) if len(top_keywords) < 3 else None # for debugging
        for k in top_keywords:
            if k[1] > 3:
                global_keywords[k[0]] += 1

    return sorted(global_keywords.items(), key=lambda x: x[1], reverse=True)

def main():
    # msg to signal that special mode is ON
    if SET_RANGE:
        if START_DATE == END_DATE:
            print(f'SET_RANGE is ON. Only tweets on {datetime.strftime(START_DATE, '%Y%m%d')} will be processed.')
        else:
            print(f'SET_RANGE is ON. Only tweets from {datetime.strftime(START_DATE, '%Y%m%d')} to {datetime.strftime(END_DATE, '%Y%m%d')} will be processed.')

    # load tweets
    tweets = load_tweets()
    print(f'🖒🖒🖒 Successfully Loaded {len(tweets)} tweets')
    print('-' * 80)


    print('📀 Processing...')
    tokenized_tweets = tokenize_tweets(tweets)
    idf, df = compute_idf(len(tweets), tokenized_tweets)
    global_keywords = find_global_keyword(tokenized_tweets, idf)
    top_keywords = [{'keyword': keyword, 'votes': votes, 'idf': idf.get(keyword, 0)} for keyword, votes in global_keywords[:GLOBAL_KEYWORDS_AMOUNT]]
    
    df_daily.append((START_DATE, df))
    for keyword in top_keywords:
        feature_names.append(keyword['keyword'])
    print('-' * 80)


    # print('keyword: votes(idf)')
    # print('-' * 60)
    # for keyword in top_keywords:
    #     print(f'{keyword['keyword']}: {keyword['votes']}({keyword['idf']})')
    # print('-' * 80)

    
    # preparing the output folder path
    print('Saving results to json file...')
    output_path= f'../data/keyword'
    if SET_RANGE:
        if START_DATE == END_DATE:
            output_path += f'/{datetime.strftime(START_DATE, '%Y%m%d')}'
        else:
            output_path += f'/{datetime.strftime(START_DATE, '%Y%m%d')}_{datetime.strftime(END_DATE, '%Y%m%d')}'
    else:
        output_path += '/all'
    os.makedirs(output_path, exist_ok=True)

    # write result.json to save keyword results
    with open(output_path + '/result.json', 'w', encoding='utf-8') as file:
        json.dump(top_keywords, file, indent=4)

    # isolate tweets by keywords
    if not DAILY_MODE:
        for idx, keyword in enumerate(top_keywords):
            tweets_by_keyword = []
            tweet_counter = 1
            for tweet, tokens in zip(tweets, tokenized_tweets):
                if keyword['keyword'] in tokens:
                    tweet['tweet_count'] = tweet_counter
                    tweet_counter += 1
                    tweets_by_keyword.append(tweet)
            
            with open(f'{output_path}/{idx+1:02d}_{keyword['keyword']}.json', 'w', encoding='utf-8') as file:
                json.dump(tweets_by_keyword, file, indent=4, ensure_ascii=False)
    print()
    print('Done.')
    print('-' * 80)

if __name__ == '__main__':
    if DAILY_MODE:
        print(f'DAILY_MODE is ON. The script will iterate from {datetime.strftime(DAILY_START_DATE, '%Y%m%d')} to {datetime.strftime(DAILY_END_DATE, '%Y%m%d')}.')
        SET_RANGE = True # just incase someone didn't read the NOTE.
        for date in rrule(DAILY, dtstart=DAILY_START_DATE, until=DAILY_END_DATE):
            START_DATE = END_DATE = date
            main()

        expansion_data = pd.read_csv(f'../data/tweets/count/estimate/{COIN_SHORT_NAME}_estimate.csv', on_bad_lines='skip')
        
        feature_names = list(set(feature_names))
        for day in range((DAILY_END_DATE - DAILY_START_DATE).days + 1):
            get_expansion_rate = expansion_data.loc[expansion_data['date'] == datetime.strftime(df_daily[day][0], '%Y-%m-%d'), 'expansion_ratio']

            if not get_expansion_rate.empty:
                expansion_rate = round(get_expansion_rate.iloc[0])
            else:
                expansion_rate = 1

            feature_vector.append([
                df_daily[day][1].get(keyword, 0) * expansion_rate for keyword in feature_names])
            print(expansion_rate)
        input()
        print(feature_vector)
        
        output_path = '../data/keyword/machine_learning/'
        os.makedirs(output_path, exist_ok=True)
        np.save(f'{output_path}/feature_vector.npy', feature_vector)

        with open(f'{output_path}/feature_name.json', 'w', encoding='utf-8') as file:
            json.dump(feature_names, file)
    else:
        main()      