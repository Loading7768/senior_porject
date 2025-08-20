import asyncio

from twikit import Client, TooManyRequests
import json
import httpx
import os

from random import randint, uniform
import random
from datetime import datetime, timedelta

import smtplib
from email.mime.text import MIMEText
from email.header import Header

import winsound

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import config_PEPE, config


# 若要直接修改 QUERY 在 198 行  並把 185 - 195 行註解
# 但 START_YEAR, START_MONTH, START_DAY 仍要填寫  為了建 json 檔名
# 而 DAY_COUNT = 1, CHANGE_MONTH = 0 即可
'''可修改參數'''
MINIMUM_TWEETS = 1000000  # 設定最少要擷取的推文數

COIN_NAME = config.COIN_NAME  # 目前要爬的 memecoin
# 
# '(officialtrump OR "official trump" OR "trump meme coin" OR "trump coin" OR trumpcoin OR $TRUMP OR "dollar trump")'
COIN_SHORT_NAME = config.COIN_SHORT_NAME  # 要當成檔案名的 memecoin 名稱

JSON_DICT_NAME = config.JSON_DICT_NAME  # 設定推文所存的 json 檔中字典的名稱

SEARCH = 'Latest'  # 在 X 的哪個欄位內搜尋 (Top, Latest, People, Media, Lists)

START_YEAR = 2025  # 開始的年份

START_MONTH = 5  # 開始的月份

START_DAY = 31  # 開始的日期

DAY_COUNT = 1  # 要連續找幾天

TRUN_ON_TIWCE_BREAK = False  # 看有沒有要當出現兩次 Rate limit reached 就馬上停止執行

# 如果不需要程式執行完成後傳 gmail 給你, 則留空字串
GMAIL = ""

PASSWORD = ""  # 帳號有啟用兩步驟驗證的話, PASSWORD 需要使用自行創建的「應用程式密碼」
'''可修改參數'''





# 定義 **異步** 函式來獲取推文
async def get_tweets(client, tweets, query):
    if tweets is None:
        # 第一次獲取推文
        print(f'{datetime.now()} - Getting tweets...')
        tweets = await client.search_tweet(query, product=SEARCH)
    else:
        # 如果已經有推文，則等待一段隨機時間後再獲取下一批推文
        base = randint(4, 7)
        wait_time = base + uniform(0.2, 1.5)  # 更自然的等待
        print(f'{datetime.now()} - Getting next tweets after {wait_time} seconds ...')
        await asyncio.sleep(wait_time)  # `await` 讓程式非同步等待
        tweets = await tweets.next()
    
    # 低機率的長時間等待 模擬人突然去忙別的事
    if random.random() < 0.05:  # 5% 機率
        break_time = randint(60, 180)  # 等 1 到 3 分鐘
        print(f'{datetime.now()} - Taking a short break for {break_time} seconds...')
        await asyncio.sleep(break_time)

    return tweets



# 用 asyncio.Lock() 來確保同一時間只有一個協程能寫入檔案
lock = asyncio.Lock()
async def save_json(data_json, filename):
    async with lock:
        with open(filename, 'w', encoding='utf-8-sig') as file:
            json.dump(data_json, file, indent=4, ensure_ascii=False)



# 先把分析資料存在 analysis_temp.txt  以防止程式中途發生錯誤  
async def write_analysis_temp(founded_count, filename, QUERY, timestamp):
    # 計算總共執行時間 (? day ? hr ? min)

    # endTime = []  # 用來記錄執行結束時間 或 目前時間
    start_time = datetime.strptime(timestamp[0][0], '%Y-%m-%d %H:%M:%S')
    
    if founded_count >= MINIMUM_TWEETS:  # 代表這輪已執行完成
        end_time = datetime.strptime(timestamp[-1][0], '%Y-%m-%d %H:%M:%S')
    else:  # 用目前的時間計算總共執行時間
        end_time = datetime.now()

    # 計算總時間差
    total_duration = end_time - start_time
    total_days = total_duration.days
    total_seconds = total_duration.seconds
    total_hours = total_seconds // 3600
    total_minutes = (total_seconds % 3600) // 60


    if founded_count > 0:
        analysisFile = '../data/tweets/analysis_temp.txt'
        with open(filename, 'r', encoding='utf-8-sig') as file:
            data_json = json.load(file)
        with open(analysisFile, 'w', encoding='utf-8-sig') as txtfile:
            txtfile.write(f"QUERY = '{QUERY}'\n")
            txtfile.write(f"SEARCH = '{SEARCH}'\n")
            txtfile.write(f"執行時間：{timestamp[0][0]} ~ {end_time.strftime('%Y-%m-%d %H:%M:%S')} ({total_days} day {total_hours} hr {total_minutes} min)\n")
            # txtfile.write(f"執行時間：{timestamp[0][0]} ~ {endTime[0]} ({totalHr} hr {totalMin} min)\n")  
            txtfile.write(f"推文數量：{data_json[JSON_DICT_NAME][-1]['tweet_count']} ({founded_count - data_json[JSON_DICT_NAME][-1]['tweet_count']} WrittingError)\n")
            txtfile.write(f"推文時間：{data_json[JSON_DICT_NAME][-1]['created_at']} ~ {data_json[JSON_DICT_NAME][0]['created_at']} (GMT+0)\n")
            txtfile.write(f"Timestamp：\n")
            for i in timestamp:
                txtfile.write(f'\t{i[0]} {i[1]}\n')
            txtfile.write('\n')



 # 發送電子郵件
async def send_email(subject, body, receiver_email):
    sender_email = GMAIL  # 你的電子郵件地址
    sender_password = PASSWORD  # 你的電子郵件密碼 (注意安全)

    # 如果有帳密有其中一個是空白 則不執行
    if sender_email != "" or sender_password != "":
        # 設定 SMTP 伺服器和端口 (以 Gmail 為例，請根據你的郵件服務提供商修改)
        smtp_server = "smtp.gmail.com"
        smtp_port = 465  # 使用 SSL

        try:
            # 建立 MIMEText 物件來設定郵件內容
            message = MIMEText(body, 'plain', 'utf-8')
            message['From'] = sender_email
            message['To'] = receiver_email
            message['Subject'] = Header(subject, 'utf-8')

            # 連接 SMTP 伺服器 (使用 SSL 加密)
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                # 登入郵件伺服器
                server.login(sender_email, sender_password)
                # 發送郵件
                server.sendmail(sender_email, receiver_email, message.as_string())
            print(f"郵件已成功發送至 {receiver_email}")

        except Exception as e:
            print(f"發送郵件失敗: {e}")
    else:
        print("帳號或密碼空白")



# 定義 **異步** 主函式
async def main():
    """主執行函式"""
    # 登入 X.com（原 Twitter）
    # 1) 直接在登入後的 X 上抓出 "auth_token", "ct0"
    # 2) 儲存並加載 Cookies 來保持登入狀態
    client = Client(language='en-US')
    client.load_cookies('tweets/cookies.json')  # 這裡 **不用 await**，因為是同步函式

    body = ""  # 用來記錄每一輪的 analysis 來傳 email

    # 設定目前是否達到此帳號抓文的上限
    TooManyRequests_bool = False
    
    global START_YEAR, START_MONTH, START_DAY

    start_date = datetime(START_YEAR, START_MONTH, START_DAY)
    for day_count in range(DAY_COUNT):
        since_date = start_date + timedelta(days=day_count)
        until_date = start_date + timedelta(days=day_count+1)

        QUERY = f'{COIN_NAME} lang:en until:{until_date.strftime("%Y-%m-%d")} since:{since_date.strftime("%Y-%m-%d")}'
        print(QUERY)

        # 更新 START_YEAR, START_MONTH, START_DAY 為最後一個 since_date
        START_YEAR = since_date.year
        START_MONTH = since_date.month
        START_DAY = since_date.day
        
        '''直接修改 QUERY'''
        # QUERY = '"official trump" lang:en until:2025-01-20 since:2025-01-19'

        # 設定推文計數
        founded_count = 0
        tweets = None

        # 將 action 都先設為有抓到推文
        action = "get"

        timestamp = []
        start_time = datetime.now()
        TooManyRequests_last = start_time
        # 設定開始時間的 timestamp
        # strftime  把 datetime 的時間用固定的格式轉成 string
        timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "Start"])

        while founded_count < MINIMUM_TWEETS:
            # 設定檔案名稱
            current_date = datetime(START_YEAR, START_MONTH, START_DAY)

            # 格式化為檔名 (可把個位數前面補零)
            date_str = current_date.strftime('%Y%m%d')  # 例：20210420

            # 如果 START_MONTH 是個位數的話 在資料夾名稱前面補 0
            if START_MONTH < 10:
                filename = f"../data/tweets/{COIN_SHORT_NAME}/{START_YEAR}/0{START_MONTH}/{COIN_SHORT_NAME}_{date_str}.json"
            else:
                filename = f"../data/tweets/{COIN_SHORT_NAME}/{START_YEAR}/{START_MONTH}/{COIN_SHORT_NAME}_{date_str}.json"
            
            try:
                tweets = await get_tweets(client, tweets, QUERY)  # `await` 確保非同步運行
                action = "get"
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
                
                # 如果在 action == "limit" 表示上一個動作也是 rate limit reached => 代表此帳號今天的抓文達上限 => 立刻 break
                # difference = datetime.now() - TooManyRequests_last
                if action == "limit" and TooManyRequests_last != start_time and TRUN_ON_TIWCE_BREAK:
                    timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"TooManyRequests - TwiceBreak"])
                    TooManyRequests_bool = True
                    print(f"{datetime.now()} - This account rate limit reached - TooManyRequests twice")
                    break

                # 紀錄本次 ToManyRequests 的時間 來當作下一次的參考時間
                TooManyRequests_last = datetime.now()

                action = "limit"
                await asyncio.sleep(wait_time)  # `await` 讓程式非同步等待
                continue
            except httpx.ConnectTimeout:
                # 如果無法正常登入 X (發生 time out) 則等待 10 分鐘（600 秒）
                print(f'{datetime.now()} - Connection timed out. Retrying in 10 minutes...')

                # 設定 timestamp
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "httpx.ConnectTimeout"])

                await asyncio.sleep(600)  # `await` 讓程式非同步等待
                continue
            except httpx.ReadTimeout: # as e 讀不出東西
                # 表示程式在嘗試從伺服器讀取資料時超時
                print(f'{datetime.now()} - Read timeout occurred. Retrying in a few seconds...')
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"httpx.ReadTimeout"])
                await asyncio.sleep(randint(5, 15)) # 等待一段時間後重試

                continue
            except Exception as e:  # 任何其他錯誤訊息
                print(f"{datetime.now()} - Search failed: {e}. Retrying in 10 seconds...")
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"{e}"])

                await asyncio.sleep(10)
                continue

            if not tweets:
                # 如果沒有推文了，結束爬取
                print(f'{datetime.now()} - No more tweets found')

                # 設定 timestamp
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "No more tweets found"])
                break


            
            '''以下為測試檔案是否正常'''
            # 將 data.json 中的資料讀到 data_json 中
            try:
                # 如果 START_MONTH 是個位數的話 在資料夾名稱前面補 0
                if START_MONTH < 10:
                    output_json_path = f"../data/tweets/{COIN_SHORT_NAME}/{START_YEAR}/0{START_MONTH}/"
                else:
                    output_json_path = f"../data/tweets/{COIN_SHORT_NAME}/{START_YEAR}/{START_MONTH}/"
                os.makedirs(output_json_path, exist_ok=True)

                with open(filename, 'r', encoding='utf-8-sig') as file:
                    data_json = json.load(file)
                
                 # 加入這行確保 key 存在
                if JSON_DICT_NAME not in data_json:
                    data_json[JSON_DICT_NAME] = []

            except (FileNotFoundError, json.JSONDecodeError):
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                data_json = {JSON_DICT_NAME: []}  # 如果檔案不存在，初始化為空字典
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
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"Test-WritingError-OSError - {e}"])

            # 抓出 data.json 中最後一筆資料的 tweet_count
            try:
                tweet_count = data_json[JSON_DICT_NAME][-1]['tweet_count']
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
                    'user_account': tweet.user.screen_name,  # 使用者帳戶名
                    'text': tweet.text,  # 推文內容
                    'created_at': tweet.created_at,  # 發布時間 (這裡是 GMT+0 的時間)
                    'retweet': tweet.retweet_count,  # 轉推數
                    'likes': tweet.favorite_count,  # 按讚數
                    'retweeted_tweet': tweet.retweeted_tweet
                }

                # 將新的 tweet 加入 data_json 裡的 dogecoin 字典中
                data_json[JSON_DICT_NAME].append(tweet_dict)


            # 抓完後一次將推文資訊寫入 data.json 檔案 (不要每一筆都 json.dump)
            # ensure_ascii=False 直接輸出原本的字元，不會轉成 Unicode 編碼
            try:
                await save_json(data_json, filename)
            except OSError as e:
                print(f"寫入檔案失敗: {e}, 檔案名: {filename}, 長度: {len(filename)}")

                # 設定 timestamp
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"WritingError-OSError - {e}"])

                continue

            # 把資料存到 analysis_temp.txt
            await write_analysis_temp(founded_count, filename, QUERY, timestamp)
                

            print(f'{datetime.now()} - Got {founded_count} tweets')

        # 爬取結束
        print(f'{datetime.now()} - Done! Got {founded_count} tweets found')


        '''判斷是否有抓完 沒有則改檔名'''
        if founded_count > 0:
            # 先將最後一則推文的時間抓出來
            earliest_str = data_json[JSON_DICT_NAME][-1]["created_at"]
            earliest_dt = datetime.strptime(earliest_str, "%a %b %d %H:%M:%S %z %Y")
            earliest_time = earliest_dt.strftime("%H:%M:%S")

            earliest_tweet_count = data_json[JSON_DICT_NAME][-1]["tweet_count"]

            # 判斷有沒有抓完 是否是 00:XX:XX  若不是則將檔名最後加上 _LatestXXX.json
            if not earliest_time.startswith("00:"):
                original_name = filename[:-5]  # 去掉 .json
                new_name = f"{original_name}_Latest{earliest_tweet_count}.json"

                # 重新命名
                os.rename(filename, new_name)
                print(f"Renamed: {filename} → {new_name}\nearliest_dt: {earliest_dt}")

                # 設定 timestamp
                timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), f"Renamed: {filename} → {new_name}"])

                filename = new_name
        '''判斷是否有抓完 沒有則改檔名'''


        # 設定開始時間的 timestamp
        timestamp.append([datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'), "End"])

        # 把資料存到 analysis_temp.txt  並算出最終的執行時間
        await write_analysis_temp(founded_count, filename, QUERY, timestamp)
        


        # 在有抓到資料的前提下 把資料存入 analysis.txt 裡
        analysis_temp = ""
        if founded_count > 0:
            analysisFile = '../data/tweets/analysis.txt'
            analysisTempFile = '../data/tweets/analysis_temp.txt'
            with open(analysisTempFile, 'r', encoding='utf-8-sig') as file:
                analysis_temp = file.read()
            with open(analysisFile, 'a', encoding='utf-8-sig') as txtfile:
                txtfile.write(analysis_temp)
        
        if analysis_temp != "":
            body = body + f"{analysis_temp}\n\n"
        else:  # 如果一整天都沒有抓到推文
            body = body + f"{QUERY}\n本日找不到推文\n\n"

        if TooManyRequests_bool:
            break
        
        base = randint(4, 7)  # 基礎時間 4s ~ 7s
        wait_time_last = base + uniform(0.2, 1.5)  # 加一點小的隨機毫秒
        wait_time_last = randint(5, 10)  # 5s ~ 10s
        print(f'{datetime.now()} - Waiting to next day after {wait_time_last} seconds ...')
        await asyncio.sleep(wait_time_last)

        # 使程式執行完成後發出提示音
        winsound.MessageBeep()
    
    # 傳送程式執行完成通知給 gamil
    subject = "Python 程式執行完成通知"
    body = f"您的 Python 程式已於 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功執行完成！\n\n" + body
    await send_email(subject, body, GMAIL)


# **執行 `main()`**，確保程式運行在 **異步模式**
asyncio.run(main())
