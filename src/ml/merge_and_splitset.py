import json
import pandas as pd
import numpy as np
import random
from scipy import sparse
import os
import pickle



'''可修改參數'''
INPUT_PATH = "../data/ml/dataset"

MIN_COUNT = 10  # 設定刪掉出現次數 <= MIN_COUNT 的關鍵詞 (column)

TOLERANCE = 100  # 資料集中誤差推文數值 (100 => +-100)

random.seed(42)  # 42 可以換成你想要的數字
'''可修改參數'''



# --- 平衡先用日期切割的後的資料 ---
def balance_sets_by_swap(train_df, val_df, test_df, target_ratios=(0.8,0.1,0.1)):
    # 計算理論值
    total_tweets = train_df['tweet_count'].sum() + val_df['tweet_count'].sum() + test_df['tweet_count'].sum()
    target_train = int(total_tweets * target_ratios[0])
    target_val   = int(total_tweets * target_ratios[1])
    target_test  = total_tweets - target_train - target_val

    # 建立 DataFrame 複製以便交換
    train = train_df.copy()
    val   = val_df.copy()
    test  = test_df.copy()

    max_iter = 10000  # 避免無限迴圈
    for _ in range(max_iter):
        # 計算目前三個 set 的推文總數
        train_sum = train['tweet_count'].sum()
        val_sum   = val['tweet_count'].sum()
        test_sum  = test['tweet_count'].sum()

        # 判斷是否所有 set 都在 tolerance 內
        if (abs(train_sum - target_train) <= TOLERANCE and
            abs(val_sum - target_val) <= TOLERANCE and
            abs(test_sum - target_test) <= TOLERANCE):
            break

        # 找一個超過理論值的 set 和小於理論值的 set    s[1] → DataFrame   s[2] → target count
        sets = [('train', train, target_train), ('val', val, target_val), ('test', test, target_test)]
        over_sets  = [s for s in sets if s[1]['tweet_count'].sum() > s[2]]
        under_sets = [s for s in sets if s[1]['tweet_count'].sum() < s[2]]

        if not over_sets or not under_sets:
            break  # 無法交換時退出

        over_name, over_df, over_target = random.choice(over_sets)
        under_name, under_df, under_target = random.choice(under_sets)

        # 隨機選日期交換，但必須 over_row['tweet_count'] > under_row['tweet_count']
        attempt = 0
        max_attempt = 100
        while attempt < max_attempt:

            # 隨機選日期交換
            over_idx = random.choice(over_df.index)
            under_idx = random.choice(under_df.index)

            # 交換兩個 row
            over_row = over_df.loc[over_idx]
            under_row = under_df.loc[under_idx]
            
            if over_row['tweet_count'] > under_row['tweet_count']:
                break
            attempt += 1
        else:
            # 如果經過 max_attempt 次還找不到符合條件的，直接跳過這輪交換
            continue

        over_df.loc[over_idx] = under_row
        under_df.loc[under_idx] = over_row

        # 更新集合
        if over_name == 'train': train = over_df
        elif over_name == 'val': val = over_df
        else: test = over_df

        if under_name == 'train': train = under_df
        elif under_name == 'val': val = under_df
        else: test = under_df

    return train, val, test



# --- 展開成每條推文一個單位 ---
def expand_by_tweet(df):
    expanded = []
    for _, row in df.iterrows():
        expanded.extend([row['date'].strftime("%Y/%m/%d")] * row['tweet_count'])
    return np.array(expanded)



# --- 列印漲跌比例 ---
def print_label_distribution(Y_train, Y_val, Y_test, COIN_SHORT_NAME, split_val=False):
    def get_stats(y):
        up = np.sum(y >= 0)
        down = np.sum(y < 0)
        total = len(y)
        return up, down, total, up / total, down / total

    train_up, train_down, train_total, train_up_ratio, train_down_ratio = get_stats(Y_train)
    if split_val:
        val_up, val_down, val_total, val_up_ratio, val_down_ratio = get_stats(Y_val)
    test_up, test_down, test_total, test_up_ratio, test_down_ratio = get_stats(Y_test)

    print(f"{COIN_SHORT_NAME} 標籤分佈：")
    print(f"  Train: 總數 {train_total}, 漲 {train_up} ({train_up_ratio:.2%}), 跌 {train_down} ({train_down_ratio:.2%})")
    if split_val:
        print(f"  Val  : 總數 {val_total}, 漲 {val_up} ({val_up_ratio:.2%}), 跌 {val_down} ({val_down_ratio:.2%})")
    print(f"  Test : 總數 {test_total}, 漲 {test_up} ({test_up_ratio:.2%}), 跌 {test_down} ({test_down_ratio:.2%})\n")



# --- 用日期為單位把資料切成 8:1:1 (train : validation : test) ---
def splitset_dates(COIN_SHORT_NAME):

    # --- 讀取每條推文的 ID .pkl ---
    with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids.pkl", "rb") as f:   # rb = read binary
        ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)]
    dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array
    
    # 如果是 bytes，要轉成 str
    if isinstance(dates[0], bytes):
        dates = dates.astype(str)

    # 轉成 datetime 方便排序
    dates_dt = pd.to_datetime(dates, format="%Y-%m-%d")

    # 統計每天出現次數
    unique_dates, counts = np.unique(dates_dt, return_counts=True)

    # 建成 dict，key 用 "YYYY/MM/DD" 格式
    date_count_dict = {pd.Timestamp(d).strftime("%Y/%m/%d"): int(c) for d, c in zip(unique_dates, counts)}

    # 儲存 JSON
    json_output_path = f"{INPUT_PATH}/coin_price"
    os.makedirs(json_output_path, exist_ok=True)
    with open(f"{json_output_path}/{COIN_SHORT_NAME}_filtered_tweet_count.json", "w", encoding="utf-8") as f:
        json.dump(date_count_dict, f, ensure_ascii=False, indent=4)


    # --- 讀 JSON ---
    with open(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_filtered_tweet_count.json", "r", encoding="utf-8") as f:
        tweet_count_dict = json.load(f)

    # --- 轉成 DataFrame ---
    df = pd.DataFrame(list(tweet_count_dict.items()), columns=['date', 'tweet_count'])
    df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")

    # --- 按日期排序或隨機打亂（這裡先打亂） ---
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 計算各資料集大小 ---
    n = len(df)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_size = n - train_size - val_size  # 剩下的給 test

    # --- 切分 ---
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # 隨機交換微調    tolerance: 誤差推文數值 (100 => +-100)
    train_df, val_df, test_df = balance_sets_by_swap(train_df, val_df, test_df)

    # 先按照 tweet_count 由大到小排序
    train_df_sorted = train_df.sort_values(by='tweet_count', ascending=False)
    val_df_sorted   = val_df.sort_values(by='tweet_count', ascending=False)
    test_df_sorted  = test_df.sort_values(by='tweet_count', ascending=False)

    csv_output_path = f"{INPUT_PATH}/split_dates"
    os.makedirs(csv_output_path, exist_ok=True)
    train_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_train_dates.csv", index=False, encoding="utf-8-sig")
    val_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_val_dates.csv", index=False, encoding="utf-8-sig")
    test_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_test_dates.csv", index=False, encoding="utf-8-sig")

    # --- 展開成每條推文一個單位 ---
    dates_train_expanded = expand_by_tweet(train_df)
    dates_val_expanded = expand_by_tweet(val_df)
    dates_test_expanded = expand_by_tweet(test_df)

    return dates_train_expanded, dates_val_expanded, dates_test_expanded



# --- 用平衡好的結果來按日期切割 X, Y，並可選擇是否要再分出 val (預設 False) ---
def splitset_XY(COIN_SHORT_NAME, split_val=False):

    # 讀取稀疏矩陣
    X = sparse.load_npz(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_X_sparse.npz")  # 二維陣列：colunm(關鍵詞) row(某天某推文) (但這裡是稀疏矩陣的格式)
    
    # 一維陣列：存放與 X row 對應的 ID
    with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids.pkl", "rb") as f:   # rb = read binary
        ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)]
    dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array

    # 讀取 price_diff.npy
    Y = np.load(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_price_diff.npy")  # shape = (總推文數,)

    # 輸出長度 確保一致性
    print(f"{COIN_SHORT_NAME}：")
    print("X.shape[0] =", X.shape[0])
    print("dates.shape[0] =", len(dates))
    print("Y.shape[0] =", len(Y), "(原始長度 未刪剪)\n")



    # 把 date 轉成 datetime 格式，方便比對
    dates_datetime = pd.to_datetime(dates)

    # 讀取三個集合的日期 (切割好的 CSV)
    train_dates = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_train_dates.csv")['date']
    val_dates   = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_val_dates.csv")['date']
    test_dates  = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_test_dates.csv")['date']

    # 轉成 datetime
    train_dates = pd.to_datetime(train_dates)
    val_dates   = pd.to_datetime(val_dates)
    test_dates  = pd.to_datetime(test_dates)

    # 找出 index   逐筆檢查 date 中的每一個值，判斷它是否在 train_dates 裡
    train_mask = dates_datetime.isin(train_dates)
    val_mask   = dates_datetime.isin(val_dates)
    test_mask  = dates_datetime.isin(test_dates)

    # 切割 X
    X_train = X[train_mask]
    X_val   = X[val_mask]
    X_test  = X[test_mask]

    # 切割 Y
    Y_train = Y[train_mask]
    Y_val   = Y[val_mask]
    Y_test  = Y[test_mask]

    # 切割 dates
    dates_train = dates[train_mask]
    dates_val   = dates[val_mask]
    dates_test  = dates[test_mask]

    if not split_val:
        # 如果不需要 validation，就把 val + test 合併
        X_test = sparse.vstack([X_val, X_test], format="csr")
        Y_test = np.concatenate([Y_val, Y_test])
        dates_test = np.concatenate([dates_val, dates_test])
        X_val, Y_val, dates_val = None, None, None  # 不返回 validation

    # 列印各個幣種 split 後 每個資料集的漲跌比例
    print_label_distribution(Y_train, Y_val, Y_test, COIN_SHORT_NAME)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, dates_train, dates_val, dates_test



# --- 將不必要的關鍵詞與推文刪除 ---
def filter_XY(X_train, X_val, X_test, Y_train, Y_val, Y_test, dates_train, dates_val, dates_test, all_vocab):
    '''重複 功能 1, 功能 2, 功能 3 直到沒有東西可以刪為止'''

    # 定義 功能 1
    def function_1(X, Y, dates):
        row_sums = np.array(X.sum(axis=1)).ravel()
        valid_rows = np.where(row_sums > 0)[0]
        invalid_rows = X.shape[0] - len(valid_rows)

        X = X[valid_rows, :]
        Y = Y[valid_rows]
        dates = dates[valid_rows]
        
        return X, Y, dates, invalid_rows, len(valid_rows)
            

    invalid_rows = -1
    delete_min_count = -1
    delete_only_test = -1

    total_delete_rows = 0
    total_delete_columns = 0

    # 若還有功能是可以刪資料的，就再繼續跑
    while invalid_rows != 0 or delete_min_count != 0 or delete_only_test != 0:

        # --- 功能 1: 刪掉沒有任何關鍵詞的推文 (刪 row) ---
        # train
        X_train, Y_train, dates_train, train_invalid_rows, train_valid_rows = function_1(X_train, Y_train, dates_train)
        
        # test
        X_test, Y_test, dates_test, test_invalid_rows, test_valid_rows = function_1(X_test, Y_test, dates_test)

        # val
        if X_val is not None:
            X_val, Y_val, dates_val, val_invalid_rows, val_valid_rows = function_1(X_val, Y_val, dates_val)

        # 計算保留、刪掉的筆數
        valid_rows = train_valid_rows + test_valid_rows
        invalid_rows = train_invalid_rows + test_invalid_rows
        if X_val is not None:
            valid_rows += val_valid_rows
            invalid_rows += val_invalid_rows
            
        # 只看 Train 的數量
        total_delete_rows += train_invalid_rows
        print("功能 1: 刪掉沒有任何關鍵詞的推文 (row):")
        print(f"\tTrain 保留 row 數量: {train_valid_rows}")
        print(f"\tTrain 刪掉 row 數量: {train_invalid_rows}\n")


        # --- 功能 2: 刪掉在 train 中出現次數 <= min_count 的關鍵詞 (刪 column) ---
        col_sums = np.array(X_train.sum(axis=0)).ravel()
        valid_cols = np.where(col_sums > MIN_COUNT)[0]

        # 每個關鍵詞的出現次數統計
        keyword_counts = {all_vocab[i]: int(col_sums[i]) for i in range(len(all_vocab))}
        stats_output_path = os.path.join(INPUT_PATH, "keyword", "keyword_counts.json")
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(keyword_counts, f, ensure_ascii=False, indent=4)

        # 開始過濾
        X_train = X_train[:, valid_cols]
        X_test  = X_test[:, valid_cols]
        if X_val is not None:
            X_val = X_val[:, valid_cols]

        filtered_vocab = [all_vocab[i] for i in valid_cols]
        delete_min_count = len(all_vocab) - len(filtered_vocab)
        total_delete_columns += delete_min_count
        print("功能 2: 刪掉出現次數 <= min_count 的關鍵詞 (column):")
        print(f"\t原始 column 數量: {len(all_vocab)}")
        print(f"\t保留 column 數量: {len(filtered_vocab)}")
        print(f"\t刪掉 column 數量: {delete_min_count}\n")

        with open(os.path.join(f"{INPUT_PATH}/keyword", f"filtered_keywords.json"), "w", encoding="utf-8") as f:
            json.dump(filtered_vocab, f, ensure_ascii=False, indent=4)  


        # --- 功能 3: 只保留 train 出現過的關鍵詞 (刪 column) --- 
        '''
        X_train.nonzero() 會回傳一個 tuple (row_idx, col_idx)：
        row_idx → 非零元素所在的 row（推文 index）。
        col_idx → 非零元素所在的 column（關鍵詞 index）。

        X_train.nonzero()[1] 取的是所有非零值的 column index。
        → 這就是「有哪些關鍵詞至少在 train 裡出現過一次」。

        np.unique(...) 把它去重，得到一個 只出現在 train 的 column 清單。
        '''

        orig_cols = X_train.shape[1]
        keep_cols = np.unique(X_train.nonzero()[1])  # train 出現過的 column index
        new_cols = len(keep_cols)

        # column 過濾
        X_train = X_train[:, keep_cols]  # [:, keep_cols] 表示「保留所有 row，但只取出 keep_cols 這些 column」。
        X_test  = X_test[:, keep_cols]
        if X_val is not None:
            X_val = X_val[:, keep_cols]
        if all_vocab is not None:
            all_vocab = [all_vocab[i] for i in keep_cols]

        delete_only_test = orig_cols - new_cols
        total_delete_columns += delete_only_test
        print("功能 3: 只保留 train 出現過的關鍵詞 (column):")
        print(f"\t原始 column 數量: {orig_cols}")
        print(f"\t保留 column 數量: {new_cols}")
        print(f"\t刪掉 column 數量: {delete_only_test}\n")

    print(f"總共刪除 {total_delete_rows} 個推文 (row), {total_delete_columns} 個關鍵詞 (column)\n")
    print(f"已輸出所有關鍵詞出現次數統計到 {stats_output_path}")
    print(f"已輸出所有被過濾的關鍵詞到 {INPUT_PATH}/keyword\n")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, dates_train, dates_val, dates_test



# --- 打亂順序 (shuffle) ---
def shuffle_XY(X, Y, dates, seed=42):
    """
    Shuffle X and Y in unison.
    X: np.ndarray 或 scipy.sparse 矩陣
    Y: np.ndarray 一維標籤
    seed: 隨機種子
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(Y.shape[0])  # 取得樣本數 indices = [0, 1, 2, ... , len(X)-1]
    rng.shuffle(indices)  # 把 indices 隨機重新排序

    # 如果是稀疏矩陣
    if sparse.issparse(X):
        X_shuffled = X[indices, :]  # 按照 indices 的順序重新排列
    else:
        X_shuffled = X[indices]

    Y_shuffled = Y[indices]  # 按照 indices 的順序重新排列

    # dates shuffle
    dates_shuffled = dates[indices]

    return X_shuffled, Y_shuffled, dates_shuffled



# --- 計算每個日期總共的數量 (用以確認是否有切正確) ---
def count_per_day(dates, dataset_name):
    """
    dates: array-like, 每條推文的日期 (str 或 np.datetime64)
    dataset_name: 用於打印
    """
    # 如果是 bytes，先轉成 str
    if isinstance(dates[0], bytes):
        dates = dates.astype(str)

    # 轉成 datetime
    dates_dt = pd.to_datetime(dates)

    # 計算每天出現次數
    date_counts = dates_dt.value_counts().sort_index()  # 按日期排序
    df_counts = pd.DataFrame({"date": date_counts.index, "tweet_count": date_counts.values})

    df_counts.to_csv(f"{INPUT_PATH}/dates_{dataset_name}_counts.csv", index=False)

    return df_counts



# --- 將三種幣種的 X, Y 合併成完整的模型輸入值 (輸出 .npy 檔) ---
def merge(DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test,
          PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test,
          TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test,
          DOGE_dates_train, DOGE_dates_val, DOGE_dates_test,
          PEPE_dates_train, PEPE_dates_val, PEPE_dates_test,
          TRUMP_dates_train, TRUMP_dates_val, TRUMP_dates_test,
          all_vocab):

    # 合併 X（稀疏矩陣用 sparse.vstack）
    X_train_list = [DOGE_X_train, PEPE_X_train, TRUMP_X_train]
    X_test_list  = [DOGE_X_test, PEPE_X_test, TRUMP_X_test]
    X_val_list   = [DOGE_X_val, PEPE_X_val, TRUMP_X_val] if DOGE_X_val is not None else []

    X_train = sparse.vstack(X_train_list, format="csr")  # np.vstack = vertical stack，把多個矩陣在「列方向」堆疊起來
    X_test  = sparse.vstack(X_test_list, format="csr")
    X_val   = sparse.vstack(X_val_list, format="csr") if X_val_list else None

    # 合併 Y
    Y_train = np.concatenate([DOGE_Y_train, PEPE_Y_train, TRUMP_Y_train])  # np.concatenate = 把多個一維陣列串接起來
    Y_test  = np.concatenate([DOGE_Y_test, PEPE_Y_test, TRUMP_Y_test])
    Y_val   = np.concatenate([DOGE_Y_val, PEPE_Y_val, TRUMP_Y_val]) if DOGE_Y_val is not None else None

    # 合併 Dates
    dates_train = np.concatenate([DOGE_dates_train, PEPE_dates_train, TRUMP_dates_train])
    dates_test  = np.concatenate([DOGE_dates_test, PEPE_dates_test, TRUMP_dates_test])
    dates_val   = np.concatenate([DOGE_dates_val, PEPE_dates_val, TRUMP_dates_val]) if DOGE_dates_val is not None else None


    # 將不必要的關鍵詞與推文刪除
    X_train, X_val, X_test, Y_train, Y_val, Y_test, dates_train, dates_val, dates_test = filter_XY(X_train, X_val, X_test, Y_train, Y_val, Y_test, dates_train, dates_val, dates_test, all_vocab)
    


    # 打亂順序
    X_train, Y_train, dates_train = shuffle_XY(X_train, Y_train, dates_train)
    X_test,  Y_test,  dates_test  = shuffle_XY(X_test, Y_test, dates_test)
    if X_val is not None:
        X_val, Y_val, dates_val = shuffle_XY(X_val, Y_val, dates_val)

    # 儲存
    sparse.save_npz(f"{INPUT_PATH}/X_train.npz", X_train)
    sparse.save_npz(f"{INPUT_PATH}/X_test.npz", X_test)
    np.save(f"{INPUT_PATH}/Y_train.npy", Y_train)
    np.save(f"{INPUT_PATH}/Y_test.npy", Y_test)

    if X_val is not None:
        sparse.save_npz(f"{INPUT_PATH}/X_val.npz", X_val)
        np.save(f"{INPUT_PATH}/Y_val.npy", Y_val)

    if dates_train is not None:
        np.savez_compressed(f"{INPUT_PATH}/dates_train.npz", arr=dates_train)
        np.savez_compressed(f"{INPUT_PATH}/dates_test.npz", arr=dates_test)
        # np.save(f"{INPUT_PATH}/dates_train.npy", dates_train)
        # np.save(f"{INPUT_PATH}/dates_test.npy", dates_test)
        if dates_val is not None:
            np.savez_compressed(f"{INPUT_PATH}/dates_val.npz", arr=dates_val)
            # np.save(f"{INPUT_PATH}/dates_val.npy", dates_val)

    # 檢查資料集維度
    assert X_train.shape[0] == len(Y_train) == len(dates_train), "Train 維度不一致!"
    assert X_test.shape[0] == len(Y_test) == len(dates_test), "Test 維度不一致!"
    if X_val is not None:
        assert X_val.shape[0] == len(Y_val) == len(dates_val), "Val 維度不一致!"


    print("Merge 完成，資料已輸出到 ../data/ml/dataset\n")

    # 列印 merge, filter 後 每個資料集的漲跌比例
    print_label_distribution(Y_train, Y_val, Y_test, "ALL")

    # 計算每個資料集中每天的推文總數
    count_per_day(dates_train, "train")
    count_per_day(dates_test, "test")
    if dates_val is not None:
        count_per_day(dates_val, "val")

    print("已將不同資料集每天的推文總數輸出為 csv 到 ../data/ml/dataset\n")





# --- 列印平衡好的結果 ---
def print_split_number(train_expanded, val_expanded, test_expanded, COIN_SHORT_NAME):
    sum = len(train_expanded) + len(val_expanded) + len(test_expanded)
    print(COIN_SHORT_NAME + "：")
    print(f"理論值 Train: {int(sum * 0.8)},                Val: {int(sum * 0.1)},                Test: {int(sum * 0.1)}")
    print(f"實際值 Train: {len(train_expanded)} ({round((len(train_expanded) / sum), 10)}), Val: {len(val_expanded)} ({round((len(val_expanded) / sum), 10)}), Test: {len(test_expanded)} ({round((len(test_expanded) / sum), 10)})\n")



def main():

    # 分別切資料集
    DOGE_dates_train_expanded, DOGE_dates_val_expanded, DOGE_dates_test_expanded = splitset_dates("DOGE")
    print_split_number(DOGE_dates_train_expanded, DOGE_dates_val_expanded, DOGE_dates_test_expanded, "DOGE")
    DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test, DOGE_dates_train, DOGE_dates_val, DOGE_dates_test = splitset_XY("DOGE")  # 若要分出 val => splitset_XY("DOGE", True)

    PEPE_dates_train_expanded, PEPE_dates_val_expanded, PEPE_dates_test_expanded = splitset_dates("PEPE")
    print_split_number(PEPE_dates_train_expanded, PEPE_dates_val_expanded, PEPE_dates_test_expanded, "PEPE")
    PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test, PEPE_dates_train, PEPE_dates_val, PEPE_dates_test = splitset_XY("PEPE")

    TRUMP_dates_train_expanded, TRUMP_dates_val_expanded, TRUMP_dates_test_expanded = splitset_dates("TRUMP")
    print_split_number(TRUMP_dates_train_expanded, TRUMP_dates_val_expanded, TRUMP_dates_test_expanded, "TRUMP")
    TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test, TRUMP_dates_train, TRUMP_dates_val, TRUMP_dates_test = splitset_XY("TRUMP")

    # 讀取所有關鍵詞的名字
    json_path = os.path.join("../data/keyword/machine_learning", "all_keywords.json")
    with open(json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    all_vocab = list(vocab)

    # 合併資料集
    merge(DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test,
          PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test,
          TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test,
          DOGE_dates_train, DOGE_dates_val, DOGE_dates_test,
          PEPE_dates_train, PEPE_dates_val, PEPE_dates_test,
          TRUMP_dates_train, TRUMP_dates_val, TRUMP_dates_test,
          all_vocab)
    

if __name__ == "__main__":
    main()