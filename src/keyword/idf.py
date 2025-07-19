'''
IDF (Inverse Document Frequency) implementation. Ignoring the TF part since
all tweets are quite short, redering the value negligible.
'''

from glob import glob
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
MIN_DF = 10 # a threshold to filter words that are too uncommon
# ----------parameters----------

# sanitize tweets
def clean_tweets(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)      # remove mentions
    text = re.sub(r'#\w+', '', text)      # remove hashtags
    # text = text.replace(COIN_SHORT_NAME.lower(), "")

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

        # file_name = json_file[27+len(COIN_SHORT_NAME)+len(sentiment)+1:]
        # print(f'🖒 Loaded {file_name}')

    return tweets

def conpute_idf(tweets, min_df):
    tokenizer = TweetTokenizer(preserve_case=False)
    STOPWORDS = set(stopwords.words('english'))
    N = len(tweets)
    df = defaultdict(int)

    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet)
        unique_tokents = set(token for token in tokens if token not in STOPWORDS and token.isalnum())
        for token in unique_tokents:
            df[token] += 1

    idf = {
        token: math.log((N+1) / (df_count+1)) + 1
        for token, df_count in df.items()
        if df_count >= min_df
    }
    return idf

def main():
    sentiments = ['positive', 'neutral', 'negative']
    for sent in sentiments:
        # load tweets
        tweets = load_tweets(sent)
        print(f'🖒🖒🖒 Successfully Loaded {len(tweets)} {sent} tweets')
        print('-' * 80)

        print('📀 Processing...')
        idf = conpute_idf(tweets, MIN_DF)
        sorted_idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)
        print('-' * 80)

        print('keyword: idf')
        print('---------------')
        for token, score in sorted_idf[:30]:
            print(f'{token}: {score:.3f}')

        print('-' * 80)
        input(f'Part {sent} done! Press enter to continue...')
        print('-' * 80)

if __name__ == '__main__':
    main()      