import json
from multiprocessing import Pool, cpu_count
from itertools import combinations
from tqdm import tqdm


'''可修改參數'''
FILENAME = "../project_vscode/data/DOGE/2021/4/DOGE_20210428_Latest654.json"  # 選擇要對哪個檔案執行

ANALYSIS_NAME = "DOGE_20210428"  # 設定要存的 txt 名稱

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱

LCS_SIMILARITY = 60  # 設定 LCS 占 Y(被比對的推文) 的比例（也就是相似度）  60 => 60%

LENGTH_RATIO = 80  # 設定 Y(被比對的推文) 的長度相對於 X(當基準的推文) 的百分比   80 => 80%
# 這是要確認兩篇推文的長度落在合理範圍內，推文 Y 的長度至少有 X 的 80% 長（不能太短）

TOKEN_OVERLAP_THRESHOLD = 30  # 設定 token_overlap 的臨界值   30 => 30%
'''可修改參數'''

txtname = f"../LCS/analysis/{ANALYSIS_NAME}.txt"

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



def save_results(results, output_path):
    with open(output_path, 'w', encoding="utf-8-sig") as filetxt:
        filetxt.write("")

    with open(output_path, 'a', encoding="utf-8-sig") as filetxt:
        for res in results:
            if res is None:
                continue
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


# 為了不要 MemoryError
# 邊用 imap_unordered 邊產生組合，不需要一次生成所有組合
def generate_pairs(tweets):
    # (1) 避免與自己比對  (2) 每一組只會比對一次
    # 生成要比對的推文組合，格式是 (i, j, tweets)
    for i, j in combinations(range(len(tweets)), 2):
        yield (i, j, tweets)


# 主程式
if __name__ == "__main__":
    # 讀入 json 檔
    with open(FILENAME, 'r', encoding="utf-8-sig") as file:
        data_json = json.load(file)

    tweets = data_json[JSON_DICT_NAME]
    print(f"載入推文數量：{len(tweets)}")

    results = []
    with Pool(processes=cpu_count()) as pool:  # Pool(processes=cpu_count()): 建立一個「工作池」來開啟多個 CPU 核心跑比對
        
        # pool.imap_unordered(compare_pair, pairs): 將 pairs 中的每組 (i, j, tweets) 傳進 compare_pair 函式處理，並行處理，誰先完成就先送回來
        for res in tqdm(pool.imap_unordered(compare_pair, generate_pairs(tweets)), 
                        total=(len(tweets) * (len(tweets) - 1)) // 2,  # 產生所有從 n 個元素中選出 2 個不重複且無順序的組合  [C(n, 2) = (n(n - 1)) / 2]
                        desc="比對中"):  # tqdm(...): 進度條
            results.append(res)  # 把處理完的結果存起來（有些會是 None，代表比對不通過被過濾掉）

    print(f"總共產生的比對結果數：{len(results)}")
    print(f"實際寫入的結果數：{sum(1 for r in results if r is not None)}")

    save_results(results, txtname)
    print(f"✅ 已輸出結果到 {txtname}")




# # filename = "d://senior_project/Kmeans/data/combined/DOGE_2021_4.json"
# with open(FILENAME, 'r', encoding="utf-8-sig") as file:
# 	data_json = json.load(file)


# # 清空指定的 txt 檔
# txtname = f"../LCS/{ANALYSIS_NAME}.txt"
# with open(txtname, 'w', encoding="utf-8-sig") as filetxt:
#     filetxt.write("")

# # (1) 避免與自己比對  (2) 每一組只會比對一次
# for i in range(len(data_json[JSON_DICT_NAME])):
#     X = data_json[JSON_DICT_NAME][i]
#     print(f"目前 X tweet_count = {X["tweet_count"]}\nX = [{repr(X["text"])[1:-1]}]")

#     for j in range(i + 1, len(data_json[JSON_DICT_NAME])):
#         Y = data_json[JSON_DICT_NAME][j]

#         # 如果比對的推文長度相差超過 LENGTH_RATIO(%) 則直接不執行 LCS
#         if (len(Y["text"]) / len(X["text"])) * 100 >= LENGTH_RATIO:
            
#             # 將 X, Y 用空格把單字分割
#             x_tokens = set(X["text"].split())
#             y_tokens = set(Y["text"].split())

#             # x_tokens & y_tokens: 有幾個詞是一樣的 (token overlap)
#             if (len(x_tokens & y_tokens) / len(y_tokens)) * 100 < TOKEN_OVERLAP_THRESHOLD:
#                 continue  # 相似度太低，跳過，不要進入 LCS 計算

#             # 回傳 LCS 的長度及字串
#             length, sequence = lcs(X["text"], Y["text"])

#             # 判斷相似度高於 LCS_SIMILARITY(%) 才輸出到 txt 檔裡
#             if (length / len(Y["text"])) * 100 >= LCS_SIMILARITY:
#                 with open(txtname, 'a', encoding="utf-8-sig") as filetxt:
#                     filetxt.write(f"X = [{repr(X["text"])[1:-1]}]\n")  # repr(): 讓 \n 保持為 \n 輸出
#                     filetxt.write(f"\tX tweet_count = [{X["tweet_count"]}]\n")
#                     filetxt.write(f"\tX username = [{X["username"]}]\n")
                    
#                     filetxt.write(f"Y = [{repr(Y["text"])[1:-1]}]\n")
#                     filetxt.write(f"\tY tweet_count = [{Y["tweet_count"]}]\n")
#                     filetxt.write(f"\tY username = [{Y["username"]}]\n")

#                     filetxt.write(f"Length of LCS: {length}\n") 
#                     filetxt.write(f"sequence of LCS = [{sequence}]\n")
#                     filetxt.write(f"Total Length: X = {len(X["text"])}, Y = {len(Y["text"])} ({(len(Y["text"]) / len(X["text"])) * 100:.2f})\n")
#                     filetxt.write(f"resemblance: {(length / len(Y["text"])) * 100:.2f}% in Y\n\n")
