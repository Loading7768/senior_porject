import json

JSON_DICT_NAME = "dogecoin"

filename = './data/DOGE/2021/5/DOGE_20210504_Latest50001_max.json'  # 修改成指定檔案
with open(filename, 'r', encoding = 'utf-8-sig') as file:
    data_json = json.load(file)

count = 0
tweetCount = 0
lastHourCount = 0  # 紀錄道上一個小時為止的所有推文數量
last_count = data_json[JSON_DICT_NAME][-1]['tweet_count']
tweetCount_perHour = []  # 記錄每小時的推文數量
hour = 23  # 目前要看哪個時段
while True:
    if int(data_json[JSON_DICT_NAME][count]['created_at'][11]) * 10 + int(data_json[JSON_DICT_NAME][count]['created_at'][12]) == hour:
        tweetCount = data_json[JSON_DICT_NAME][count]['tweet_count']
        count += 1
    else:
        tweetCount_perHour.append(tweetCount - lastHourCount)  # 每個小時分別的推文數量
        # tweetCount_perHour.append(tweetCount)  # 每個小時累加的數量
        hour -= 1
        lastHourCount = tweetCount

    if data_json[JSON_DICT_NAME][count]['tweet_count'] == last_count:
        tweetCount_perHour.append(tweetCount - lastHourCount + 1)  # 每個小時分別的推文數量
        # tweetCount_perHour.append(last_count)  # 每個小時累加的數量
        break
    
for i in range(len(tweetCount_perHour)):
    print(f"{23 - i} hr - {tweetCount_perHour[i]}")

