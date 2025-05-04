import json
from multiprocessing import Pool, cpu_count
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict


'''可修改參數'''
FILENAME = "../Kmeans/data/clustered/clustered_30_DOGE_20210428.json"  # 選擇要對哪個檔案執行
# "../project_vscode/data/DOGE/2021/4/DOGE_20210428_Latest654.json"
# 

ANALYSIS_NAME = "DOGE_20210428_test"  # 設定要存的 txt 名稱

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱

LCS_SIMILARITY = 60  # 設定 LCS 占 Y(被比對的推文) 的比例（也就是相似度）  60 => 60%

LENGTH_RATIO = 80  # 設定 Y(被比對的推文) 的長度相對於 X(當基準的推文) 的百分比   80 => 80%
# 這是要確認兩篇推文的長度落在合理範圍內，推文 Y 的長度至少有 X 的 80% 長（不能太短）

TOKEN_OVERLAP_THRESHOLD = 30  # 設定 token_overlap 的臨界值   30 => 30%

IS_CLUSTERED = True  # 設定是否要用有分群的檔案來比對
'''可修改參數'''

if IS_CLUSTERED:
    txtname = f"../LCS/analysis/{ANALYSIS_NAME}_clustered.txt"
    json_output_path = f"../LCS/analysis/{ANALYSIS_NAME}_clustered.json"
else:
    txtname = f"../LCS/analysis/{ANALYSIS_NAME}.txt"
    json_output_path = f"../LCS/analysis/{ANALYSIS_NAME}.json"

def lcs(X, Y):
    m = len(X)
    n = len(Y)

    # 建立 DP 陣列
    L = [[0] * (n + 1) for _ in range(m + 1)]

    # 填入 DP 陣列的值
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # 回溯找出 LCS 字串
    i, j = m, n
    lcs_str = []

    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs_str.append(X[i - 1])
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 回溯結果是反過來的，要反轉
    lcs_str.reverse()

    return L[m][n], ''.join(lcs_str)



def compare_pair(args):
    i, j, tweets = args
    X = tweets[i]
    Y = tweets[j]

    # 如果比對的推文長度相差超過 LENGTH_RATIO(%) 則直接不執行 LCS
    if (len(Y["text"]) / len(X["text"])) * 100 < LENGTH_RATIO:
        return None

    # 將 X, Y 用空格把單字分割
    x_tokens = set(X["text"].split())
    y_tokens = set(Y["text"].split())

    # x_tokens & y_tokens: 有幾個詞是一樣的 (token overlap)
    if (len(x_tokens & y_tokens) / len(y_tokens)) * 100 < TOKEN_OVERLAP_THRESHOLD:
        return None  # 相似度太低，跳過，不要進入 LCS 計算

    # 回傳 LCS 的長度及字串
    length, sequence = lcs(X["text"], Y["text"])
    # 判斷相似度高於 LCS_SIMILARITY(%) 才輸出到 txt 檔裡
    if (length / len(Y["text"])) * 100 < LCS_SIMILARITY:
        return None

    return {
        "X": X,
        "Y": Y,
        "length": length,
        "sequence": sequence
    }



# 為了不要 MemoryError
# 邊用 imap_unordered 邊產生組合，不需要一次生成所有組合
def generate_pairs(tweets):
    # (1) 避免與自己比對  (2) 每一組只會比對一次
    # 生成要比對的推文組合，格式是 (i, j, tweets)
    for i, j in combinations(range(len(tweets)), 2):
        yield (i, j, tweets)


def write_txt_result(filetxt, res):
    X, Y = res["X"], res["Y"]

    filetxt.write(f"X = [{repr(X['text'])[1:-1]}]\n")  # repr(): 讓 \n 保持為 \n 輸出
    filetxt.write(f"\tX tweet_count = [{X['tweet_count']}]\n")
    filetxt.write(f"\tX username = [{X['username']}]\n")

    filetxt.write(f"Y = [{repr(Y['text'])[1:-1]}]\n")
    filetxt.write(f"\tY tweet_count = [{Y['tweet_count']}]\n")
    filetxt.write(f"\tY username = [{Y['username']}]\n")

    filetxt.write(f"Length of LCS: {res['length']}\n")
    filetxt.write(f"sequence of LCS = [{repr(res['sequence'])[1:-1]}]\n")
    filetxt.write(f"Total Length: X = {len(X['text'])}, Y = {len(Y['text'])} "
                  f"({(len(Y['text']) / len(X['text'])) * 100:.2f})\n")
    filetxt.write(f"resemblance: {(res['length'] / len(Y['text'])) * 100:.2f}% in Y\n\n")



def write_json_result(res, cluster_id=None):
    X, Y = res["X"], res["Y"]
    data = {
        "X_text": X["text"],
        "X_tweet_count": X["tweet_count"],
        "X_username": X["username"],
        "Y_text": Y["text"],
        "Y_tweet_count": Y["tweet_count"],
        "Y_username": Y["username"],
        "lcs_length": res["length"],
        "lcs_sequence": res["sequence"],
        "X_length": len(X["text"]),
        "Y_length": len(Y["text"]),
        "Y_length_percent_of_X": round((len(Y["text"]) / len(X["text"])) * 100, 2),
        "lcs_similarity_percent_in_Y": round((res["length"] / len(Y["text"])) * 100, 2)
    }
    if cluster_id is not None:
        data["cluster_id"] = cluster_id
    return data



def process_tweet_group(tweets_group, json_output, cluster_id=None):
    writed_compare = 0  # 實際寫入的結果數

    with Pool(processes=cpu_count()) as pool:  # Pool(processes=cpu_count()): 建立一個「工作池」來開啟多個 CPU 核心跑比對

        # pool.imap_unordered(compare_pair, pairs): 將 pairs 中的每組 (i, j, tweets) 傳進 compare_pair 函式處理，並行處理，誰先完成就先送回來
        for res in tqdm(pool.imap_unordered(compare_pair, generate_pairs(tweets_group)),  # tqdm(...): 進度條
                        total=(len(tweets_group) * (len(tweets_group) - 1)) // 2,  # 產生所有從 n 個元素中選出 2 個不重複且無順序的組合  [C(n, 2) = (n(n - 1)) / 2]
                        desc=f"比對中 (Cluster {cluster_id})" if cluster_id is not None else "比對中"):
            
            if res is None:  # 如果是 None，代表比對不通過被過濾掉
                continue  # 不寫進 txt 檔

            writed_compare += 1
            json_output.append(write_json_result(res, cluster_id=cluster_id))
            # 寫入 JSON
            with open(json_output_path, 'w', encoding='utf-8') as f_json:
                json.dump(json_output, f_json, indent=4, ensure_ascii=False)

            write_txt_result(filetxt, res)

    return writed_compare




# 主程式
if __name__ == "__main__":
    # 讀入 json 檔
    with open(FILENAME, 'r', encoding="utf-8-sig") as file:
        data_json = json.load(file)

    tweets = data_json[JSON_DICT_NAME]
    print(f"載入推文數量：{len(tweets)}")

    # 先把 txt 檔裡清空
    with open(txtname, 'w', encoding="utf-8-sig") as filetxt:
        filetxt.write("")



    json_output = []  # 用來儲存所有比對結果
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
                total_compare += process_tweet_group(cluster_tweets, json_output, cluster_id=cluster_id)
        else:
            # 如果是沒有分類過的檔案 直接呼叫 process_tweet_group 來執行比對
            total_compare = process_tweet_group(tweets, json_output)

    

    print(f"✅ 已儲存 JSON 結果到 {json_output_path}")
    print(f"實際寫入的全部結果數：{total_compare}")
    print(f"✅ 已輸出結果到 {txtname}")