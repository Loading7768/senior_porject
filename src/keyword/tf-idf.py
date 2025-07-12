'''
attempt to find keywords of tweets under different sentiments
'''

from glob import glob
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

# ----------parameters----------

# ----------parameters----------

# load tweets
def load_tweets(sentiment):
    tweets = []

    json_files = glob(f'../data/sentiment/{COIN_SHORT_NAME}/*/*/{sentiment}/*')
    for json_file in json_files:
        with open(json_file, 'r', encoding="utf-8") as file:
            data = json.load(file)

            for tweet in data:
                tweets.append(tweet.get('text'))

        print(f'Loaded {json_file.replace(f'../data/sentiment/{COIN_SHORT_NAME}/*/*/{sentiment}', '')}')

    return tweets

def main():
    sentiments = ['positive', 'neutral', 'negative']
    for sent in sentiments:
        # load tweets
        tweets = load_tweets(sent)
        print('-' * 80)

        # tf-idf processing
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df= 0.5,
            min_df= 3,
            ngram_range=(1,2),
            max_features= 10000
        )

        result = vectorizer.fit_transform(tweets)

        # filter top results
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sum = result.sum(axis=0)
        tfidf_sum = np.asarray(tfidf_sum).flatten()

        print('Top 30 keywords:')
        top_indices = tfidf_sum.argsort()[::-1][:30]
        for i in top_indices:
            print(f'{feature_names[i]:<20} {tfidf_sum[i]:.4f}')

        print('-' * 80)


        # system 'pause'
        input('Press enter to continue...')

if __name__ == '__main__':
    main()