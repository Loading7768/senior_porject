import json
import os
# from multiprocessing import Pool, cpu_count
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords




'''可修改參數'''
FOLDER_PATH = "../temp_data"  # 選擇要對哪個資料夾執行
# "../Kmeans/data/clustered/"
# "../project_vscode/data/spammer/04"

OUTPUT_FOLDER_NAME = "20210416"  # 設定要儲存到的資料夾名稱   ex. "../LCS/analysis/{OUTPUT_FOLDER_NAME}/"

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱

DICE_COEFFICIENT = 70  # 設定 Dice 算出來的結果門檻值（也就是相似度）  60 => 60%

LENGTH_RATIO = 80  # 設定 Y(被比對的推文) 的長度相對於 X(當基準的推文) 的百分比   80 => 80%
# 這是要確認兩篇推文的長度落在合理範圍內，推文 Y 的長度至少有 X 的 80% 長（不能太短）

IS_CLUSTERED = False  # 設定是否要用有分群的檔案來比對

# 第一次使用時下載
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
'''可修改參數'''






# 取得英文停用詞集合
stop_words = set(stopwords.words('english'))

# 計算 Dice
def dice(X, Y):
    # 將 X Y 用 NLTK 的 word_tokenize 做單字切割
    # 利用 lower() 把所有字母都變小寫
    x_tokens = word_tokenize(X.lower(), language='english')
    y_tokens = word_tokenize(Y.lower(), language='english')

    # 移除 stopwords
    # 利用 word.isalnum() 過濾掉標點符號   ex. "!".isalnum() → False（有驚嘆號）   "dogecoin".isalnum() → True
    x_tokens = [word for word in x_tokens if word.isalnum() and (word not in stop_words)]  # 這裡是把每個 token 分別拿出來檢查
    y_tokens = [word for word in y_tokens if word.isalnum() and (word not in stop_words)]

    # 轉成集合以便後續比對
    # 但如果有重複的單字 會只算成一次
    x_set = set(x_tokens)
    y_set = set(y_tokens)

    # 在 set 的格式中 & 為聯集
    overlap = len(x_set & y_set)
    total = len(x_set) + len(y_set)

    if total == 0:
        return 0.0, x_tokens, y_tokens  # 避免除以零

    # 計算 Dice Coefficient = (2 * |X & Y|) / (|X| + |Y|)
    dice_score = (2 * overlap) / total
    return dice_score, x_tokens, y_tokens

    


def compare_pair(args):
    i, j, tweets = args
    X = tweets[i]
    Y = tweets[j]

    # 如果比對的推文長度相差超過 LENGTH_RATIO(%) 則直接不執行 LCS
    length_ratio_X = (len(Y["text"]) / len(X["text"]))
    length_ratio_Y = (len(X["text"]) / len(Y["text"]))
    if length_ratio_X * 100 < LENGTH_RATIO or length_ratio_Y * 100 < LENGTH_RATIO:
        return None

    # 回傳 Dice Coefficient
    dice_coefficient, x_tokens, y_tokens = dice(X["text"], Y["text"])
    # 判斷相似度高於 DICE_COEFFICIENT(%) 才輸出到 txt 檔裡
    if dice_coefficient * 100 < DICE_COEFFICIENT:
        return None

    return {
        "X": X,
        "Y": Y,
        "X_token": ", ".join(x_tokens),
        "Y_token": ", ".join(y_tokens),
        "length_ratio_X": length_ratio_X,
        "length_ratio_Y": length_ratio_Y,
        "dice_coefficient": dice_coefficient
    }



# 為了不要 MemoryError
# 邊用 imap_unordered 邊產生組合，不需要一次生成所有組合
def generate_pairs(tweets):
    # (1) 避免與自己比對  (2) 每一組只會比對一次
    # 生成要比對的推文組合，格式是 (i, j, tweets)
    for i, j in combinations(range(len(tweets)), 2):
        yield (i, j, tweets)

    # return ((i, j, tweets) for i, j in combinations(range(len(tweets)), 2))


def write_txt_result(filetxt, res):
    X, Y = res["X"], res["Y"]

    filetxt.write(f"X = [{repr(X['text'])[1:-1]}]\n")  # repr(): 讓 \n 保持為 \n 輸出
    filetxt.write(f"X_token = [{res['X_token']}]\n")
    filetxt.write(f"\tX tweet_count = [{X['tweet_count']}]\n")
    filetxt.write(f"\tX username = [{X['username']}]\n")

    filetxt.write(f"Y = [{repr(Y['text'])[1:-1]}]\n")
    filetxt.write(f"Y_token = [{res['Y_token']}]\n")
    filetxt.write(f"\tY tweet_count = [{Y['tweet_count']}]\n")
    filetxt.write(f"\tY username = [{Y['username']}]\n")

    filetxt.write(f"Total Length: X = {len(X['text'])}, Y = {len(Y['text'])} "
                  f"(Y / X = {res['length_ratio_X'] * 100:.2f}  X / Y = {res['length_ratio_Y'] * 100:.2f})\n")
    filetxt.write(f"Dice Coefficient: {res['dice_coefficient'] * 100:.2f}% \n\n")



def write_json_result(res, cluster_id=None):
    X, Y = res["X"], res["Y"]
    data = {
        "X_text": X["text"],
        "X_token": res["X_token"],
        "X_tweet_count": X["tweet_count"],
        "X_username": X["username"],
        "Y_text": Y["text"],
        "Y_token": res["Y_token"],
        "Y_tweet_count": Y["tweet_count"],
        "Y_username": Y["username"],
        "X_length": len(X["text"]),
        "Y_length": len(Y["text"]),
        "Y_length_percent_of_X": round(res['length_ratio_X'] * 100, 2),
        "X_length_percent_of_Y": round(res['length_ratio_Y'] * 100, 2),
        "dice_coefficient": round(res['dice_coefficient'] * 100, 2)  # round( , 2) 輸出到小數點後第二位
    }
    if cluster_id is not None:
        data["cluster_id"] = cluster_id
    return data



def process_tweet_group(tweets_group, json_output, json_output_path, cluster_id=None, filetxt=None):
    writed_compare = 0  # 實際寫入的結果數
    fail_count = 0

    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    # map(compare_pair, pairs): 將 pairs 中的每組 (i, j, tweets) 傳進 compare_pair 函式處理，並行處理，誰先完成就先送回來
    for res in tqdm(map(compare_pair, generate_pairs(tweets_group)),  # tqdm(...): 進度條
                    total=(len(tweets_group) * (len(tweets_group) - 1)) // 2,  # 產生所有從 n 個元素中選出 2 個不重複且無順序的組合  [C(n, 2) = (n(n - 1)) / 2]
                    desc=f"比對中 (Cluster {cluster_id})" if cluster_id is not None else "比對中"):
        
        if res is None:  # 如果是 None，代表比對不通過被過濾掉
            continue  # 不寫進 txt, json 檔

        writed_compare += 1
        json_output.append(write_json_result(res, cluster_id=cluster_id))

        # 寫入 txt
        write_txt_result(filetxt, res)


    # 全部比對完再寫入 JSON
    try:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[ERROR] Failed to write after fail_count = {fail_count}: {e}")
        fail_count += 1
    

    return writed_compare



# 🧠 主程式入口：處理整個資料夾
if __name__ == "__main__":
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]
    print(f"📂 總共找到 {len(all_files)} 個檔案要處理")

    # 先清空 robottxt.txt
    robottxt = f"./robot_account/{OUTPUT_FOLDER_NAME}.txt"
    with open(robottxt, "w", encoding="utf-8-sig") as robotfile:
        robotfile.write("")

    for file in all_files:
        # run_for_file(os.path.join(FOLDER_PATH, file))
        filepath = os.path.join(FOLDER_PATH, file)
        filename = os.path.basename(filepath)  # ex: DOGE_20210428.json
        analysis_name = os.path.splitext(filename)[0]  # ex: DOGE_20210428

        # 設定 txtname, json_output_path 的名稱
        txtname = f"./analysis/{OUTPUT_FOLDER_NAME}/{analysis_name}.txt"
        json_output_path = f"./analysis/{OUTPUT_FOLDER_NAME}/{analysis_name}.json"

        # 確認是否有輸出時需使用的資料夾
        output_folder_path = f"./analysis/{OUTPUT_FOLDER_NAME}/"
        os.makedirs(output_folder_path, exist_ok=True)

        # 讀入 json 檔
        with open(filepath, 'r', encoding="utf-8-sig") as file:
            data_json = json.load(file)

        tweets = data_json[JSON_DICT_NAME]
        print(f"\n📄 正在處理檔案：{filename}，共 {len(tweets)} 筆推文")

        # 先把 txt 檔裡清空
        with open(txtname, 'w', encoding="utf-8-sig") as filetxt:
            filetxt.write("")

        json_output = []  # 用來儲存所有比對結果

        # 先把輸出的 json 檔裡清空
        with open(json_output_path, 'w', encoding='utf-8-sig') as f_json:
            json.dump(json_output, f_json, indent=4, ensure_ascii=False)



        total_compare = 0  # 計算總共寫入的結果數

        with open(txtname, 'w', encoding="utf-8-sig") as filetxt:
            if IS_CLUSTERED:
                # defaultdict: Python 的一種特殊字典
                # 當你存取一個不存在的 key 時，它會自動建立對應的預設值  ex. clusters["0"] 若原本不存在，會自動被建立成 []（空 list）
                clusters = defaultdict(list)

                for tweet in tweets:
                    clusters[tweet["cluster"]].append(tweet)  # 把推文加入對應的群集  ex. clusters[0].append(推文)

                # .items() 來一次取得 key（cluster_id）與對應的 value（cluster_tweets，一個 list）
                for cluster_id, cluster_tweets in clusters.items():
                    if len(cluster_tweets) < 2:
                        continue  # 不需要比對

                    filetxt.write(f"cluster {cluster_id}, 共 {len(cluster_tweets)} 筆\n")

                    # 呼叫 process_tweet_group 來執行比對，並回傳當前 Cluster 的實際寫入數量
                    total_compare += process_tweet_group(cluster_tweets, json_output, json_output_path, cluster_id=cluster_id, filetxt=filetxt)
            else:
                # 如果是沒有分類過的檔案 直接呼叫 process_tweet_group 來執行比對
                total_compare = process_tweet_group(tweets, json_output, json_output_path, filetxt=filetxt)

        print()
        print(f"✅ 已儲存 JSON 結果到 {json_output_path}")
        print(f"實際寫入的全部結果數：{total_compare}")
        print(f"✅ 已輸出結果到 {txtname}")



        # 建立一個字典記錄每個帳號有多少重複推文
        repetitive_counts = defaultdict(int)

        # 當你從 Dice 對比結果中抓出重複推文時
        # 你可以記錄帳號出現的次數
        for tweet in json_output:
            X_user = tweet["X_username"]
            Y_user = tweet["Y_username"]
            repetitive_counts[X_user] += 1
            repetitive_counts[Y_user] += 1


        robottxt = f"./robot_account/{OUTPUT_FOLDER_NAME}.txt"
        # 印出出現次數大於 10 的帳號，符合的話就輸出到 txt 檔中
        print()
        with open(robottxt, "a", encoding="utf-8-sig") as robotfile:
            robotfile.write(f"{filename}\n")
            for user, count in sorted(repetitive_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 10:
                    robotfile.write(f"🤖 疑似洗版帳號：{user}，重複出現次數：{count}\n")
            robotfile.write("\n")