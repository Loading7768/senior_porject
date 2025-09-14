'''
separate tweets into normal user or bots

Bots:
    - Posting too frequently
    - Tweets too similar
    - Regular posting at specific time
'''

from pathlib import Path
import json
from datetime import datetime
from glob import glob
import os
from tqdm import tqdm

import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME, JSON_DICT_NAME

def load_data():
    TWEET_PATH = f'../data/tweets/{COIN_SHORT_NAME}/*/*/*'
    SPAMMER_LIST_PATH = Path(f'../data/spammer/{COIN_SHORT_NAME}/{COIN_SHORT_NAME}_spammers.txt')
    DICE_LIST_PATH = Path(f'../data/dice/{COIN_SHORT_NAME}/{COIN_SHORT_NAME}_robot_list.txt')

    tweet_files = glob(TWEET_PATH)

    bot_list = set()
    with open(SPAMMER_LIST_PATH, 'r',encoding="utf-8") as file:
        spammer = {line.strip() for line in file if line.strip()}
        bot_list.update(spammer)

    with open(DICE_LIST_PATH, 'r', encoding="utf-8") as file:
        robots = {line.strip() for line in file if line.strip()}
        bot_list.update(robots)

    return tweet_files, bot_list

def filter_and_save(tweet_files, bot_list):
    TWEET_PARENT_PATH = f'../data/tweets/{COIN_SHORT_NAME}/*/*'
    sub_folders = glob(TWEET_PARENT_PATH)
    OUT_NORMAL_PATH = f'../data/filtered_tweets/noraml_tweets'
    OUT_BOT_PATH = f'../data/filtered_tweets/bot_tweets'
    for sf in sub_folders:
        os.makedirs(sf.replace(f'../data/tweets', OUT_NORMAL_PATH), exist_ok=True)
        os.makedirs(sf.replace(f'../data/tweets', OUT_BOT_PATH), exist_ok=True)

    for tf in tqdm(tweet_files, desc=f'Filtering...'):
        with open(tf, 'r', encoding="utf-8-sig") as file:
            data = json.load(file)
            tweets = data.get(JSON_DICT_NAME, [])

        normal_tweets = []
        bot_tweets = []
        for t in tweets:
            bot_tweets.append(t) if t.get('username') in bot_list else normal_tweets.append(t)
    
        for idx, t in enumerate(normal_tweets, start=1):
            t['tweet_count'] = idx

        for idx, t in enumerate(bot_tweets, start=1):
            t['tweet_count'] = idx

        with open(tf.replace('../data/tweets', OUT_NORMAL_PATH).replace('.json', '_normal.json'), 'w', encoding='utf-8-sig') as file:
            json.dump({JSON_DICT_NAME: normal_tweets}, file, ensure_ascii=False, indent=4)

        with open(tf.replace('../data/tweets', OUT_BOT_PATH).replace('.json', '_bot.json'), 'w', encoding='utf-8-sig') as file:
            json.dump({JSON_DICT_NAME: bot_tweets}, file, ensure_ascii=False, indent=4)

def main():
    tweet_files, bot_list = load_data()
    filter_and_save(tweet_files, bot_list)

if __name__ == '__main__':
    main()
