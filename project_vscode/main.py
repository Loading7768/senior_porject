import asyncio
from twikit import Client, TooManyRequests
# import time
from datetime import datetime
from random import randint
import json
import os
import httpx



'''可修改參數'''
MINIMUM_TWEETS = 500  # 設定最少要擷取的推文數

QUERY = 'dogecoin lang:en until:2025-03-02 since:2025-03-01'  # 可以改為其他關鍵字來搜尋不同內容

COIN_NAME = "dogecoin"  # 目前要爬的 memecoin

SEARCH = 'Latest'  # 在 X 的哪個欄位內搜尋 (Top, Latest, People, Media, Lists)
'''可修改參數'''



timestamp = []

# 定義 **異步** 函式來獲取推文
async def get_tweets(client, tweets):
    if tweets is None:
        # 第一次獲取推文
        print(f'{datetime.now()} - Getting tweets...')
        tweets = await client.search_tweet(QUERY, product=SEARCH)
    else:
        # 如果已經有推文，則等待一段隨機時間後再獲取下一批推文
        wait_time = randint(5, 10)  # 5s ~ 10s
        print(f'{datetime.now()} - Getting next tweets after {wait_time} seconds ...')
        await asyncio.sleep(wait_time)  # `await` 讓程式非同步等待
        tweets = await tweets.next()

    return tweets


# 定義 **異步** 主函式
async def main():
    """主執行函式"""
    # 登入 X.com（原 Twitter）
    # 1) 直接在登入後的 X 上抓出 "auth_token", "ct0"
    # 2) 儲存並加載 Cookies 來保持登入狀態
    client = Client(language='en-US')
    client.load_cookies('cookies.json')  # 這裡 **不用 await**，因為是同步函式

    # 設定推文計數
    founded_count = 0
    tweets = None

    # 設定開始時間的 timestamp
    # strftime  把 datetime 的時間用固定的格式轉成 string
    timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "Start"])

    while founded_count < MINIMUM_TWEETS:
        try:
            tweets = await get_tweets(client, tweets)  # `await` 確保非同步運行
        except TooManyRequests as e:
            # 如果 MINIMUM_TWEETS 太大導致達到 API 限制，等待直到限制解除
            rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
            print(f'{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}')
            wait_time = (rate_limit_reset - datetime.now()).total_seconds()

            # 設定 timestamp
            timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"TooManyRequests - Got {founded_count} tweets"])
            
            # 計算總共需等待時間 (? min ? sec)
            totalMin = int(wait_time) // 60
            totalSec = int(wait_time) % 60

            timestamp.append(["Waiting until ", f"{datetime.strftime(rate_limit_reset, '%Y-%m-%d %H:%M:%S')} ({totalMin} min {totalSec} sec)"])

            await asyncio.sleep(wait_time)  # `await` 讓程式非同步等待
            continue
        except httpx.ConnectTimeout:
            # 如果無法正常登入 X (發生 time out) 則等待 10 分鐘（600 秒）
            print(f'{datetime.now()} - Connection timed out. Retrying in 10 minutes...')

            # 設定 timestamp
            timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "httpx.ConnectTimeout"])

            await asyncio.sleep(600)  # `await` 讓程式非同步等待
            continue

        if not tweets:
            # 如果沒有推文了，結束爬取
            print(f'{datetime.now()} - No more tweets found')
            break


        
        '''以下為測試檔案是否正常'''
        filename = os.path.join("data", "data.json")  # 避免無效字元

        # 將 data.json 中的資料讀到 data_json 中
        try:
            with open(filename, 'r', encoding='utf-8-sig') as file:
                data_json = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data_json = {COIN_NAME: []}  # 如果檔案不存在，初始化為空字典
            with open(filename, 'w', encoding='utf-8-sig') as file:
                json.dump(data_json, file, indent=4, ensure_ascii=False)

            # 設定 timestamp
            timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "FileNotFoundError"])

        # 測試寫入
        try:
            with open(filename, 'w', encoding='utf-8-sig') as file:
                json.dump(data_json, file, indent=4, ensure_ascii=False)
            print(f"測試-成功寫入 {filename}")
        except OSError as e:
            print(f"測試-寫入檔案失敗: {e}")

            # 設定 timestamp
            timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"Test-WrittingError-OSError - {e}"])

        # 抓出 data.json 中最後一筆資料的 tweet_count
        try:
            tweet_count = data_json[COIN_NAME][-1]['tweet_count']
        except (IndexError, KeyError):
            tweet_count = 0
        '''以上為測試檔案是否正常'''



        # 改成用 json 格式，最大的欄位是每個幣的名字，用 dictionary 包起來
        for tweet in tweets:
            tweet_count += 1
            founded_count += 1

            tweet_dict = {
                'tweet_count': tweet_count,
                'username': tweet.user.name,  # 使用者名稱
                'text': tweet.text,  # 推文內容
                'created_at': tweet.created_at,  # 發布時間 (這裡是 GMT+0 的時間)
                'retweet': tweet.retweet_count,  # 轉推數
                'likes': tweet.favorite_count  # 按讚數
            }

            # 將新的 tweet 加入 data_json 裡的 dogecoin 字典中
            data_json[COIN_NAME].append(tweet_dict)

            # 將推文資訊寫入 data.json 檔案
            # ensure_ascii=False 直接輸出原本的字元，不會轉成 Unicode 編碼
            try:
                with open(filename, 'w', encoding='utf-8-sig') as file:
                    json.dump(data_json, file, indent=4, ensure_ascii=False)
            except OSError as e:
                print(f"寫入檔案失敗: {e}")

                # 設定 timestamp
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"WrittingError-OSError - {e}"])

                continue

        print(f'{datetime.now()} - Got {founded_count} tweets')

    # 爬取結束
    print(f'{datetime.now()} - Done! Got {founded_count} tweets found')

    # 設定開始時間的 timestamp
    timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "End"])

    # 計算總共執行時間 (? hr ? min)
    # timestamp[?][0][11] timestamp[?][0][12] 是 hr   timestamp[?][0][14] timestamp[?][0][15] 是 min
    hrStart = int(timestamp[0][0][11]) * 10 + int(timestamp[0][0][12])
    minStart = int(timestamp[0][0][14]) * 10 + int(timestamp[0][0][15])
    hrEnd = int(timestamp[-1][0][11]) * 10 + int(timestamp[-1][0][12])
    minEnd = int(timestamp[-1][0][14]) * 10 + int(timestamp[-1][0][15])
    carry = False  # 如果 min 有進位的話  totalHr -= 1

    if minEnd < minStart:
        totalMin = (60 - minStart) + minEnd
        carry = True
    else:
        totalMin = minEnd - minStart

    if hrEnd < hrStart:
        totalHr = (24 - hrStart) + hrEnd
    else:
        totalHr = hrEnd - hrStart
    
    if carry:
        totalHr -= 1
    


    # 把資料存入 analysis.txt 裡
    analysisFile = '../analysis.txt'
    with open(filename, 'r', encoding='utf-8-sig') as file:
        data_json = json.load(file)
    with open(analysisFile, 'a', encoding='utf-8-sig') as txtfile:
        txtfile.write(f"QUERY = '{QUERY}'\n")
        txtfile.write(f"SEARCH = '{SEARCH}'\n")
        txtfile.write(f'執行時間：{timestamp[0][0]} ~ {timestamp[-1][0]} ({totalHr} hr {totalMin} min)\n')  
        txtfile.write(f'推文數量：{data_json[COIN_NAME][-1]['tweet_count']} ({founded_count - data_json[COIN_NAME][-1]['tweet_count']} WrittingError)\n')
        txtfile.write(f'推文時間：{data_json[COIN_NAME][-1]['created_at']} ~ {data_json[COIN_NAME][0]['created_at']} (GMT+0)\n')
        txtfile.write(f'Timestamp：\n')
        for i in timestamp:
            txtfile.write(f'\t{i[0]} {i[1]}\n')
        txtfile.write('\n')

# **執行 `main()`**，確保程式運行在 **異步模式**
asyncio.run(main())
