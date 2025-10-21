'''
attempt to find keywords of tweets under different sentiments
'''

from glob import glob
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import csv

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

# ----------parameters----------

# ----------parameters----------

# sanitize tweets
def clean_tweets(text):
    text = re.sub(r'[A-Z]{3,5}', '', text)  # remove coin short names
    text = text.lower()
    text = re.sub(r'http\S+', '', text)     # remove URLs
    text = re.sub(r'@\w+', '', text)        # remove mentions
    text = re.sub(r'#\w+', '', text)        # remove hashtags
    text = re.sub(r'\d', '', text)          # remove numbers
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

        # print(f'Loaded {json_file.replace(f'../data/sentiment/{COIN_SHORT_NAME}/*/*/{sentiment}', '')}')

    return tweets

def custom_tokenizer(text):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)
    tokens = tokenizer.tokenize(text)
    return [token for token in tokens if token.isalnum()]

def main():
    sentiments = ['positive', 'neutral', 'negative']
    for sent in sentiments:
        # load tweets
        tweets = load_tweets(sent)
        print(f'🖒🖒🖒 Successfully Loaded {len(tweets)} {sent} tweets')
        print('-' * 80)   


        # tf-idf processing
        print('📀 Processing...')
        vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer,
            stop_words='english',
            max_df= 0.95,
            min_df= 3,
            ngram_range=(1,2),
            max_features= 10000
        )

        result = vectorizer.fit_transform(tweets)
        print('-' * 80)


        # filter top results
        terms = vectorizer.get_feature_names_out()
        global_tfidf = defaultdict(int)
        for i in range(result.shape[0]):
            row = result[i].toarray().flatten()
            top_indicies = row.argsort()[-10:][::-1]
            for idx in top_indicies:
                global_tfidf[terms[idx]] += 1
            
        sorted_global = sorted(global_tfidf.items(), key=lambda x: x[1], reverse=True)

        print('term: count')
        print('-----------')
        for term, count in sorted_global[:30]:
            print(f'{term}: {count}')


        # system 'pause'
        print('-' * 80)
        input(f'Batch {sent} done! Press enter to continue...')
        print('-' * 80)

if __name__ == '__main__':
    main()