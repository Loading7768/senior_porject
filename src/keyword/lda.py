'''
attempt to find keywords of tweets under different sentiments
'''

from glob import glob
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)      # remove mentions
    text = re.sub(r'#\w+', '', text)      # remove hashtags
    text = text.replace(COIN_SHORT_NAME.lower(), "")

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

        file_name = json_file[27+len(COIN_SHORT_NAME)+len(sentiment)+1:]
        print(f'🖒 Loaded {file_name}')

    return tweets

def display_topics(model, feature_names, top_n=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f'\n ⛓️‍💥 Topic ({topic_idx + 1})')
        top_indices = topic.argsort()[::-1][:top_n]
        for i in top_indices:
            print(f'{feature_names[i]} ({topic[i]:.2f})')

def main():
    sentiments = ['positive', 'neutral', 'negative']
    for sent in sentiments:
        # load tweets
        tweets = load_tweets(sent)

        print()
        print(f'🖒🖒🖒 Successfully Loaded {len(tweets)} {sent} tweets')
        print('-' * 80)
        print()

        # lda
        print('📀 Processing...')

        vectorizer = CountVectorizer(
            stop_words='english',
            max_df=0.95,
            min_df=5,
            max_features=10000
        )
        X = vectorizer.fit_transform(tweets)
        feature_names = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
           n_components=5,
           max_iter=10,
           learning_method='online',
           random_state=42,
           n_jobs=1 
        )
        lda.fit(X)

        print()
        print('-' * 80)


        # print results
        display_topics(lda, feature_names)

        print()
        input(f'Part {sent} done! Press enter to continue...')
        print('-' * 80)
        print()
        

if __name__ == '__main__':
    main()