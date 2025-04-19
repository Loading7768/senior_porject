# 如果 tweet_count 不是從 1 開始的話可以使用
import json

JSON_DICT_NAME = "realDonaldTrump"

filename = './data/realDonaldTrump.json'  # 修改成指定檔案
with open(filename, 'r', encoding='utf-8-sig') as file:
    data_json = json.load(file)

count = 0
last_text = data_json[JSON_DICT_NAME][-1]['text']
while True:
    number = data_json[JSON_DICT_NAME][count]['tweet_count']
    data_json[JSON_DICT_NAME][count]['tweet_count'] = count + 1
    
    if data_json[JSON_DICT_NAME][count]['text'] == last_text:
        break

    count += 1

with open(filename, 'w', encoding='utf-8-sig') as file:
    json.dump(data_json, file, indent=4, ensure_ascii=False)