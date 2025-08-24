'''
Reads all gather tweets of an entire month,
make a table of counts and final hour for every date,
print in markdown format for sharing in chats.
'''

from datetime import datetime
from glob import glob
import json
import re

#----------------------------------------------------------------parameters---------------------------------------------------------------
COIN_SHORT_NAME = 'TRUMP'
YEAR = '2025'
MONTH = '08'
DICTIONARY_NAME = '(officialtrump OR "official trump" OR "trump meme coin" OR "trump coin" OR trumpcoin OR $TRUMP OR "dollar trump")'
#---------------------------------------------------------------parameters----------------------------------------------------------------

path_list = glob(f'../data/tweets/{COIN_SHORT_NAME}/{YEAR}/{MONTH}/*')

table = [('date', 'count', 'final_hour')]
for path in path_list:
    with open(path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
        data = data.get(DICTIONARY_NAME, [])
        date = re.search(r'\d{8}', path)
        lastHr = data[-1]['created_at'][11:13] + 'hr'
        table.append((date.group(), len(data), lastHr))

print('```python\n[')
for t in table:
    print(f'\t{t},')
print(']\n```')