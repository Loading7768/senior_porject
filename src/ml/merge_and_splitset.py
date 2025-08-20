import json
import pandas as pd
import numpy as np
import random
from scipy import sparse
import os



random.seed(42)  # 42 可以換成你想要的數字

# --- 平衡先用日期切割的後的資料 ---
def balance_sets_by_swap(train_df, val_df, test_df, target_ratios=(0.8,0.1,0.1), tolerance=100):
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
        if (abs(train_sum - target_train) <= tolerance and
            abs(val_sum - target_val) <= tolerance and
            abs(test_sum - target_test) <= tolerance):
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



# --- 用日期為單位把資料切成 8:1:1 (train : validation : test) ---
def splitset_dates(COIN_SHORT_NAME):
    # --- 讀 JSON ---
    with open(f"../data/ml/dataset/coin_price/{COIN_SHORT_NAME}_current_tweet_count.json", "r", encoding="utf-8") as f:
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
    train_df, val_df, test_df = balance_sets_by_swap(train_df, val_df, test_df, tolerance=100)

    # 先按照 tweet_count 由大到小排序
    train_df_sorted = train_df.sort_values(by='tweet_count', ascending=False)
    val_df_sorted   = val_df.sort_values(by='tweet_count', ascending=False)
    test_df_sorted  = test_df.sort_values(by='tweet_count', ascending=False)

    csv_output_path = "../data/ml/dataset/split_dates"
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
    X = sparse.load_npz(f"../data/ml/dataset/keyword/{COIN_SHORT_NAME}_X_sparse.npz")  # 二維陣列：colunm(關鍵詞) row(某天某推文) (但這裡是稀疏矩陣的格式)
    dates = np.load(f"../data/ml/dataset/keyword/{COIN_SHORT_NAME}_dates.npy")  # 一維陣列：存放與 X row 對應的日期

    # 讀取 price_diff.npy
    Y = np.load(f"../data/ml/dataset/coin_price/{COIN_SHORT_NAME}_price_diff.npy")  # shape = (總推文數,)

    # 把 date 轉成 datetime 格式，方便比對
    dates = pd.to_datetime(dates, format="%Y-%m-%d")

    # 讀取三個集合的日期 (切割好的 CSV)
    train_dates = pd.read_csv(f"../data/ml/dataset/split_dates/{COIN_SHORT_NAME}_train_dates.csv")['date']
    val_dates   = pd.read_csv(f"../data/ml/dataset/split_dates/{COIN_SHORT_NAME}_val_dates.csv")['date']
    test_dates  = pd.read_csv(f"../data/ml/dataset/split_dates/{COIN_SHORT_NAME}_test_dates.csv")['date']

    # 轉成 datetime
    train_dates = pd.to_datetime(train_dates, format="%Y-%m-%d")
    val_dates   = pd.to_datetime(val_dates, format="%Y-%m-%d")
    test_dates  = pd.to_datetime(test_dates, format="%Y-%m-%d")

    # 找出 index   逐筆檢查 date 中的每一個值，判斷它是否在 train_dates 裡
    train_mask = dates.isin(train_dates)
    val_mask   = dates.isin(val_dates)
    test_mask  = dates.isin(test_dates)

    # 切割 X
    X_train = X[train_mask]
    X_val   = X[val_mask]
    X_test  = X[test_mask]

    # 切割 Y
    Y_train = Y[train_mask]
    Y_val   = Y[val_mask]
    Y_test  = Y[test_mask]

    if not split_val:
        # 如果不需要 validation，就把 val + test 合併
        X_test = sparse.vstack([X_val, X_test], format="csr")
        Y_test = np.concatenate([Y_val, Y_test])
        X_val, Y_val = None, None  # 不返回 validation

    return X_train, X_val, X_test, Y_train, Y_val, Y_test



# --- 打亂順序 (shuffle) ---
def shuffle_xy(X, Y, seed=42):
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

    return X_shuffled, Y_shuffled



# --- 將三種幣種的 X, Y 合併成完整的模型輸入值 (輸出 .npy 檔) ---
def merge(DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test,
          PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test,
          TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test):

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

    # 打亂順序
    X_train, Y_train = shuffle_xy(X_train, Y_train)
    X_test,  Y_test  = shuffle_xy(X_test, Y_test)
    if X_val is not None:
        X_val, Y_val = shuffle_xy(X_val, Y_val)

    # 儲存
    sparse.save_npz("../data/ml/dataset/X_train.npz", X_train)
    sparse.save_npz("../data/ml/dataset/X_test.npz", X_test)
    np.save("../data/ml/dataset/Y_train.npy", Y_train)
    np.save("../data/ml/dataset/Y_test.npy", Y_test)

    if X_val is not None:
        sparse.save_npz("../data/ml/dataset/X_val.npz", X_val)
        np.save("../data/ml/dataset/Y_val.npy", Y_val)

    print("Merge 完成，資料已輸出到 ../data/ml/dataset")

    # return X_train, X_val, X_test, Y_train, Y_val, Y_test



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
    DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test = splitset_XY("DOGE")  # 若要分出 val => splitset_XY("DOGE", True)

    PEPE_dates_train_expanded, PEPE_dates_val_expanded, PEPE_dates_test_expanded = splitset_dates("PEPE")
    print_split_number(PEPE_dates_train_expanded, PEPE_dates_val_expanded, PEPE_dates_test_expanded, "PEPE")
    PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test = splitset_XY("PEPE")

    TRUMP_dates_train_expanded, TRUMP_dates_val_expanded, TRUMP_dates_test_expanded = splitset_dates("TRUMP")
    print_split_number(TRUMP_dates_train_expanded, TRUMP_dates_val_expanded, TRUMP_dates_test_expanded, "TRUMP")
    TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test = splitset_XY("TRUMP")

    # 合併資料集
    merge(DOGE_X_train, DOGE_X_val, DOGE_X_test, DOGE_Y_train, DOGE_Y_val, DOGE_Y_test,
          PEPE_X_train, PEPE_X_val, PEPE_X_test, PEPE_Y_train, PEPE_Y_val, PEPE_Y_test,
          TRUMP_X_train, TRUMP_X_val, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_val, TRUMP_Y_test)
    

if __name__ == "__main__":
    main()