'''
IDF (Inverse Document Frequency) implementation. Ignoring the TF part since
all tweets are quite short, words are unlikley to show up more than once,
redering the value negligible in the same tweet.
'''

from glob import glob
from datetime import datetime
import bisect
import json
import re
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import math

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

# ----------parameters----------
'''
You can modify SET_RANGE to set whether date range is applied,
'False' will use every file in the folder,
'True' will use the data within START_DATE ~ END_DATE(inclusive).

Set the date using datetime(Y, D, M), no need to fill in extra 0s.
'''
SET_RANGE = True
START_DATE = datetime(2025, 1, 18)
END_DATE = datetime(2025, 1, 31)
# ----------parameters----------

# sanitize tweets
def clean_tweets(text):
    # deprecated since removing elements prior might interfere with tokenization
    # text = re.sub(r'http\S+', '', text)   # remove URLs
    # text = re.sub(r'@\w+', '', text)      # remove mentions
    # text = re.sub(r'#\w+', '', text)      # remove hashtags
    # text = text.replace(COIN_SHORT_NAME.lower(), "")
    
    text = text.lower()
    text = text.strip()

    return text

# load tweets
def load_tweets(sentiment):
    tweets = []

    json_files = glob(f'../data/sentiment/{COIN_SHORT_NAME}/*/*/{sentiment}/*')
    if not json_files:
        print(f'No files found in {sentiment} directory.')
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
            return []
        json_files = json_files[left-1:right]

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding="utf-8") as file:
                data = json.load(file)

                for tweet in data:
                    tweets.append(clean_tweets(tweet.get('text')))
        except(json.JSONDecodeError, FileNotFoundError) as e:
            print(f'Error loading {json_file}: {e}')
            continue

    return tweets

def tokenize_tweets(tweets):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    STOPWORDS = set(stopwords.words('english'))
    tokenized_tweets = []

    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet)
        unique_tokens = set(token for token in tokens if token not in STOPWORDS and token.isalnum())
        tokenized_tweets.append(unique_tokens)

    return tokenized_tweets

def compute_idf(N, tokenized_tweets):
    '''
    Args:
        N: corpus(tweets) size
        tokenized_tweets: a list of tokens from each tweets
    '''
    df = defaultdict(int)

    for tweet_tokens in tokenized_tweets:
        for token in tweet_tokens:
            df[token] += 1

    idf = {token: math.log((N) / (df_count+1)) + 1 for token, df_count in df.items()}

    return idf

def find_global_keyword(tokenized_tweets, idf, top_n=3):
    '''
    find gloabl keyword by casting votes with top n idf from each tweets as candidates.
    Args:
        top_n (int): determines the number of candidates pulled from each tweets.
    '''
    global_keywords = defaultdict(int)

    for tweet_tokens in tokenized_tweets:
        token_idfs = [(token, idf.get(token, 0)) for token in tweet_tokens]
        top_keywrods = sorted(token_idfs, key=lambda x: x[1], reverse=True)[:top_n]
        
        # print(top_keywrods) if len(top_keywrods) < 3 else None # for debugging
        for k in top_keywrods:
            if k[1] > 3:
                global_keywords[k[0]] += 1

    return sorted(global_keywords.items(), key=lambda x: x[1], reverse=True)

def main():
    if SET_RANGE:
        print(f'SET_RANGE is on. Only tweets from {datetime.strftime(START_DATE, '%Y%m%d')} to {datetime.strftime(END_DATE, '%Y%m%d')} will be processed.')

    sentiments = ['positive', 'neutral', 'negative']
    for sent in sentiments:
        # load tweets
        tweets = load_tweets(sent)
        print(f'🖒🖒🖒 Successfully Loaded {len(tweets)} {sent} tweets')
        print('-' * 80)

        print('📀 Processing...')
        tokenized_tweets = tokenize_tweets(tweets)
        idf = compute_idf(len(tweets), tokenized_tweets)
        global_keywords = find_global_keyword(tokenized_tweets, idf)
        print('-' * 80)

        print('keyword: votes(idf)')
        print('---------------')
        for keyword, votes in global_keywords[:30]:
            print(f'{keyword}: {votes}({idf.get(keyword, 0)})')

        print('-' * 80)
        input(f'Part {sent} done! Press enter to continue...')
        print('-' * 80)

if __name__ == '__main__':
    main()      