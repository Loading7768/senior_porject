'''
IDF (Inverse Document Frequency) implementation. Ignoring the TF part since
all tweets are quite short, words are unlikley to show up more than once,
redering the value negligible in the same tweet.
'''

from glob import glob
import datetime
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
START_DATE = ''
END_DATE = ''
# ----------parameters----------

# sanitize tweets
def clean_tweets(text):
    text = text.lower()
    # text = re.sub(r'http\S+', '', text)   # remove URLs
    # text = re.sub(r'@\w+', '', text)      # remove mentions
    # text = re.sub(r'#\w+', '', text)      # remove hashtags
    # text = text.replace(COIN_SHORT_NAME.lower(), "")
    text = text.strip()

    return text

# load tweets
def load_tweets(sentiment):
    tweets = []

    json_files = glob(f'../data/sentiment/{COIN_SHORT_NAME}/*/*/{sentiment}/*')
    for json_file in json_files:
        with open(json_file, 'r', encoding="utf-8") as file:
            data = json.load(file)

            for tweet in data:
                tweets.append(clean_tweets(tweet.get('text')))

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

    idf = {token: math.log((N+1) / (df_count+1)) + 1 for token, df_count in df.items()}

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