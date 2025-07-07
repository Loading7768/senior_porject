'''
This script gets the tweet count up to 7 days prior using X API v2

!!! Keep in mind some error status also count as a request. (Fucking stupid. I know.) !!!
'''

import requests
import os
import json

from pathlib import Path
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import config

'''parameters'''
TOKEN = config.X_API_V2_ACCESS_TOKEN
COIN_SHORT_NAME = config.COIN_SHORT_NAME
QUERY = config.QUERY_STRING
START_TIME = '2025-07-01T00:00:00Z' #YYYY-MM-DDTHH:mm:ssZ
END_TIME = '2025-07-06T23:59:59Z' #YYYY-MM-DDTHH:mm:ssZ
'''parameters'''

endpoint = "https://api.twitter.com/2/tweets/counts/recent"
querystring = {
    "granularity":"day",
    "query":QUERY,
    "start_time":START_TIME,
    "end_time":END_TIME
    }
headers = {"Authorization": f"Bearer {TOKEN}"}
response = requests.request('GET', endpoint, headers=headers, params=querystring)

if response.ok:
    OUTPUT = "../data/tweets/count"
    os.makedirs(OUTPUT, exist_ok=True)
    with open(f"{OUTPUT}/{COIN_SHORT_NAME}_{START_TIME}_{END_TIME}_count.json", 'w', encoding='utf-8-sig') as file:
        json.dump(response.text, file, indent=4)
else:
    print(response.text)
