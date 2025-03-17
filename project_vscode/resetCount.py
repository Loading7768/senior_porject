# 如果 tweet_count 不是從 1 開始的話可以使用
import json

filename = './data/DOGE/2025/DOGE_20250310.json'  # 修改成指定檔案
with open(filename, 'r', encoding='utf-8-sig') as file:
    data_json = json.load(file)

count = 0
while True:
    number = data_json['dogecoin'][count]['tweet_count']
    data_json['dogecoin'][count]['tweet_count'] = count + 1
    
    if number == 4666:  # 修改成最後一個 tweet_count
        break

    count += 1

with open(filename, 'w', encoding='utf-8-sig') as file:
    json.dump(data_json, file, indent=4, ensure_ascii=False)