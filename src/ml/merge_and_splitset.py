import json
import pandas as pd
import numpy as np
import random
from scipy import sparse
import os
import pickle
from tqdm import tqdm
# from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum, PULP_CBC_CMD


'''可修改參數'''
INPUT_PATH = "../data/ml/dataset"

# MIN_COUNT = 10  # 設定刪掉出現次數 <= MIN_COUNT 的關鍵詞 (column)

TOLERANCE = 100  # 資料集中誤差推文數值 (100 => +-100)

# LABEL_COUNT = 5  # Y 中有幾組 labels

random.seed(42)  # 42 可以換成你想要的數字

IS_FILTERED = True  # 看是否有分 normal 與 bot
'''可修改參數'''

SUFFIX = "" if IS_FILTERED else "_non_filtered"




# --- 展開成每條推文一個單位 ---
def expand_by_tweet(df):
    expanded = []
    for _, row in df.iterrows():
        expanded.extend([row['date'].strftime("%Y-%m-%d")] * row['tweet_count'])
    return np.array(expanded)



# --- 列印漲跌比例 ---
def print_label_distribution(Y_train, Y_test, COIN_SHORT_NAME):
    def get_stats(y):
        up = np.sum(y >= 0)
        down = np.sum(y < 0)
        total = len(y)
        return up, down, total, up / total, down / total

    train_up, train_down, train_total, train_up_ratio, train_down_ratio = get_stats(Y_train)
    test_up, test_down, test_total, test_up_ratio, test_down_ratio = get_stats(Y_test)

    print(f"{COIN_SHORT_NAME}{SUFFIX} 標籤分佈：")
    print(f"  Train: 總數 {train_total}, 漲 {train_up} ({train_up_ratio:.2%}), 跌 {train_down} ({train_down_ratio:.2%})")
    print(f"  Test : 總數 {test_total}, 漲 {test_up} ({test_up_ratio:.2%}), 跌 {test_down} ({test_down_ratio:.2%})\n")



def categorize_array_multi(Y, t1=-0.0590, t2=-0.0102, t3=0.0060, t4=0.0657, ids=None):
    """
    Y: np.ndarray, shape = (num_labels,), 價格變化率
    t1, t2: 五元分類閾值，百分比
    """

    # 五元分類
    labels = np.full_like(Y, 2, dtype=int)  # 預設持平
    labels[Y <= t1] = 0  # 大跌
    labels[(Y > t1) & (Y <= t2)] = 1  # 跌
    labels[(Y >= t3) & (Y < t4)] = 3  # 漲
    labels[Y >= t4] = 4  # 大漲

    if ids is not None:
        # 找出 Y==0 的索引
        zero_idx = np.where(Y == 0)[0]
        # 只取對應的 ids
        dates_is_0 = set((ids[i][0], ids[i][1]) for i in zero_idx)
        if len(dates_is_0) > 0:
            print(f"共有 {len(dates_is_0)} 天 Y==0")
            for id in sorted(dates_is_0):
                print(id)

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正 TRUMP 前兩天的價格相同 第一天設為大漲

    return labels



def balance_category_proportion_swap(train_df, test_df, category_id, target_ratio=0.8, tolerance=0.001):
    """
    通過交換 train_df 和 test_df 中的行來調整特定類別的推文總數比例。
    交換要求：必須是相同 category 和相同 weekday_num。
    """
    
    # 創建副本以避免修改原始 DataFrame (雖然在函數外部已經是副本，但這是良好的實踐)
    current_train_df = train_df.copy()
    current_test_df = test_df.copy()
    
    # 迭代調整迴圈
    iteration = 0
    MAX_ITER = 200  # 設置最大迭代次數以防止無限迴圈
    
    while iteration < MAX_ITER:
        iteration += 1
        
        # 1. 計算當前總和和比例
        # 篩選出當前類別 k 的數據
        train_k = current_train_df[current_train_df['category'] == category_id]
        test_k = current_test_df[current_test_df['category'] == category_id]
        
        train_sum_k = train_k['tweet_count'].sum()
        test_sum_k = test_k['tweet_count'].sum()
        total_sum_k = train_sum_k + test_sum_k
        
        # 避免除以零
        if total_sum_k == 0:
            print(f"  類別 {category_id} 總數為零，跳過調整。")
            break
            
        current_ratio = train_sum_k / total_sum_k
        ratio_diff = current_ratio - target_ratio
        
        # 2. 檢查是否滿足目標比例
        if abs(ratio_diff) <= tolerance:
            print(f"  ✅ 類別 {category_id} 比例 {current_ratio:.4f} 滿足目標 {target_ratio:.4f} (差異 {ratio_diff:.4f})，耗時 {iteration} 輪。")
            break
            
        # 3. 決定調整方向
        if ratio_diff > tolerance:
            # Train 過多 (需要 Train 總和下降)。尋找 Train(大) 和 Test(小) 進行交換
            
            # 從 Train 中選出要移除的行 (tweet_count 應該較大)
            # 從 Test 中選出要換入的行 (tweet_count 應該較小)
            # 必須匹配 category 和 weekday_num
            
            # 將 Train/Test 數據按 weekday_num 分組，以便在同一星期內找到匹配對象
            train_k_grouped = train_k.groupby('weekday_num')
            test_k_grouped = test_k.groupby('weekday_num')
            
            best_swap_net_change = -np.inf # 追蹤最大的 net_change (負值，因為我們要下降 Train 總和)
            best_train_idx = None
            best_test_idx = None
            
            # 遍歷每個星期
            for weekday_num in range(7):
                if weekday_num in train_k_grouped.groups and weekday_num in test_k_grouped.groups:
                    train_week_data = train_k_grouped.get_group(weekday_num)
                    test_week_data = test_k_grouped.get_group(weekday_num)
                    
                    # 尋找 Train(大) 和 Test(小) 的配對
                    # 為了使 Train 總和下降最多，我們應該找：
                    # 1. Train 中最大的 tweet_count (row_out)
                    # 2. Test 中最小的 tweet_count (row_in)
                    
                    if not train_week_data.empty and not test_week_data.empty:
                    
                        if(iteration <= 100):
                            # 找到 Train 中最大的行
                            train_row_out = train_week_data.loc[train_week_data['tweet_count'].idxmax()]
                            # 找到 Test 中最小的行
                            test_row_in = test_week_data.loc[test_week_data['tweet_count'].idxmin()]
                        else:
                            try:
                                train_row_out = train_week_data.loc[train_week_data['tweet_count'].nlargest(2).index[1]]
                                test_row_in = test_week_data.loc[test_week_data['tweet_count'].nsmallest(2).index[1]]
                            except:
                                train_row_out = train_week_data.loc[train_week_data['tweet_count'].idxmax()]
                                test_row_in = test_week_data.loc[test_week_data['tweet_count'].idxmin()]
                        
                        
                        
                        # 計算交換後的淨變化： Net Change = tweet_count(in) - tweet_count(out)
                        net_change = test_row_in['tweet_count'] - train_row_out['tweet_count']
                        
                        # 只有在淨變化為負值 (Train 總數下降) 且比當前最佳交換更好時才更新
                        if net_change < 0 and net_change > best_swap_net_change:
                            best_swap_net_change = net_change
                            best_train_idx = train_row_out.name
                            best_test_idx = test_row_in.name

            
            # 執行交換
            if best_train_idx is not None:
                # 取得要交換的兩行
                row_out = current_train_df.loc[best_train_idx].copy()
                row_in = current_test_df.loc[best_test_idx].copy()
                
                # 交換：
                current_train_df.loc[best_train_idx] = row_in
                current_test_df.loc[best_test_idx] = row_out

                print(f"  -> 類別 {category_id} (Train:{current_ratio:.4f}): 交換 Train {row_out['date'].strftime('%Y-%m-%d')} ({row_out['tweet_count']}) 和 Test {row_in['date'].strftime('%Y-%m-%d')} ({row_in['tweet_count']})。 淨變動: {best_swap_net_change}")

            else:
                print(f"  ❌ 類別 {category_id} Train 過多，但找不到合適的交換對象 (Train(大) > Test(小))。停止。")
                break
                
        else: # ratio_diff < -tolerance
            # Test 過多 (需要 Train 總和上升)。尋找 Train(小) 和 Test(大) 進行交換
            
            # 邏輯與 Train 過多相反
            train_k_grouped = train_k.groupby('weekday_num')
            test_k_grouped = test_k.groupby('weekday_num')
            
            best_swap_net_change = np.inf # 追蹤最小的 net_change (正值，因為我們要上升 Train 總和)
            best_train_idx = None
            best_test_idx = None
            
            for weekday_num in range(7):
                if weekday_num in train_k_grouped.groups and weekday_num in test_k_grouped.groups:
                    train_week_data = train_k_grouped.get_group(weekday_num)
                    test_week_data = test_k_grouped.get_group(weekday_num)
                    
                    if not train_week_data.empty and not test_week_data.empty:
                        # 尋找 Train(小) 和 Test(大) 的配對
                        # 1. Train 中最小的 tweet_count (row_out)
                        # 2. Test 中最大的 tweet_count (row_in)
                        
                        train_row_out = train_week_data.loc[train_week_data['tweet_count'].idxmin()]
                        test_row_in = test_week_data.loc[test_week_data['tweet_count'].idxmax()]
                        
                        net_change = test_row_in['tweet_count'] - train_row_out['tweet_count']
                        
                        # 只有在淨變化為正值 (Train 總數上升) 且比當前最佳交換更好時才更新
                        if net_change > 0 and net_change < best_swap_net_change:
                            best_swap_net_change = net_change
                            best_train_idx = train_row_out.name
                            best_test_idx = test_row_in.name
            
            # 執行交換
            if best_train_idx is not None:
                row_out = current_train_df.loc[best_train_idx].copy()
                row_in = current_test_df.loc[best_test_idx].copy()
                
                # 交換：
                current_train_df.loc[best_train_idx] = row_in
                current_test_df.loc[best_test_idx] = row_out

                print(f"  <- 類別 {category_id} (Train:{current_ratio:.4f}): 交換 Train {row_out['date'].strftime('%Y-%m-%d')} ({row_out['tweet_count']}) 和 Test {row_in['date'].strftime('%Y-%m-%d')} ({row_in['tweet_count']})。 淨變動: {best_swap_net_change}")
            else:
                print(f"  ❌ 類別 {category_id} Test 過多，但找不到合適的交換對象 (Test(大) > Train(小))。停止。")
                break
                
    # 如果達到最大迭代次數，也停止
    if iteration == MAX_ITER:
        print(f"  ⚠️ 類別 {category_id} 達到最大迭代次數 {MAX_ITER}，可能無法收斂。")


    # 階段性檢查
    print("\n======== 階段性檢查 ========")
    # 重新計算並輸出結果
    final_train_sum = current_train_df.groupby('category')['tweet_count'].sum()
    final_test_sum = current_test_df.groupby('category')['tweet_count'].sum()
    final_combined_sum_df = pd.DataFrame({'train_sum': final_train_sum, 'test_sum': final_test_sum}).fillna(0)
    final_combined_sum_df['total_sum'] = final_combined_sum_df['train_sum'] + final_combined_sum_df['test_sum']
    final_combined_sum_df['train_ratio'] = final_combined_sum_df['train_sum'] / final_combined_sum_df['total_sum']

    print(final_combined_sum_df[['train_sum', 'total_sum', 'train_ratio']])

    print()
    print("len(current_train_df):", len(current_train_df))
    print("len(current_test_df):", len(current_test_df))
    print("current_train_df['tweet_count'].sum():", current_train_df['tweet_count'].sum())
    print("current_test_df['tweet_count'].sum():", current_test_df['tweet_count'].sum())

    return current_train_df, current_test_df



def split_by_week(df):
    current_date_pointer = 0  # 紀錄目前是正在看哪一天
    next_date_pointer = 0  # 記錄下一個 pointer 要指的位置 (current_date_pointer += next_date_pointer)
    run_count = 1  # 紀錄目前跑到第幾次 ( 4 次要給 Train， 1 次要給 Test)

    current_all_week_date = []  # 放當周全部的日期
    train_list, test_list = [], []  # 放最終要回傳的 df，裡面存 train、test 實際的資料

    # 先把 'date' 欄位轉成 datetime 格式
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    # print(df.info())
    # print(df)

    # 當如果 df 裡面沒有資料的時候跳出迴圈
    while(current_date_pointer < len(df)):
        start_week = df.iloc[current_date_pointer]['weekday_num']
        start_date = pd.to_datetime(df.iloc[current_date_pointer]['date'], format="%Y-%m-%d")
        # print("start_date:", start_date)

        if start_week != 6:  # 如果第一天不是 星期日(6) 就找出當周的星期日是幾號
            current_date = pd.to_datetime(df.iloc[current_date_pointer]['date'], format="%Y-%m-%d")
            start_date = current_date - pd.Timedelta(days = (start_week + 1))  # 如果是 星期一(0) 當周起始日期就 -1，依此類推

        for i in range(7):  # 把當周其他原本的日期找出來 並且 只留真正有資料的日期
            current_date = start_date + pd.Timedelta(days = i)
            # print(current_date)
            # print(df['date'])
            if current_date in df['date'].values:
                current_all_week_date.append(start_date + pd.Timedelta(days = i))
                next_date_pointer += 1
        # print(current_all_week_date)


        if current_all_week_date: # 確保有資料才處理
            # 篩選步驟：使用 isin() 進行高效篩選
            filtered_df = df[df['date'].isin(current_all_week_date)].copy()
            
            if run_count % 5 != 0:
                # 這是 Train Data
                train_list.append(filtered_df)
                # print(f"第 {run_count} 週：加入 Train Data，共 {len(filtered_df)} 筆資料")
                # print(train_list)
            else:
                # 這是 Test Data (每 5 次一次)
                test_list.append(filtered_df)
                # print(f"第 {run_count} 週：加入 Test Data，共 {len(filtered_df)} 筆資料")
        # input("pause...")

        current_all_week_date = []
        current_date_pointer += next_date_pointer
        next_date_pointer = 0
        run_count += 1
    

    # 將 train_list, test_list 轉成 Dataframe
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    print("train_df:\n", train_df)
    print("test_df:\n", test_df)
    print("\ntrain_df['tweet_count'].sum():", train_df['tweet_count'].sum())
    print("test_df['tweet_count'].sum():", test_df['tweet_count'].sum())
    print()


    # 計算訓練集 (train_df) 中每個 category 的總 tweet_count
    train_category_sum = train_df.groupby('category')['tweet_count'].sum()
    test_category_sum = test_df.groupby('category')['tweet_count'].sum()

    # 將兩個 Series 合併到一個 DataFrame，並處理可能不重疊的類別 (雖然在您的例子中是重疊的)
    combined_sum_df = pd.DataFrame({
        'train_sum': train_category_sum,
        'test_sum': test_category_sum
    }).fillna(0) # 如果某個類別只出現在其中一個集合，用 0 填充

    # 計算每個類別的總和 (Train + Test)
    combined_sum_df['total_sum'] = combined_sum_df['train_sum'] + combined_sum_df['test_sum']

    # Train 的比例： train_sum / total_sum
    combined_sum_df['train_ratio'] = combined_sum_df['train_sum'] / combined_sum_df['total_sum']
    combined_sum_df['test_ratio'] = combined_sum_df['test_sum'] / combined_sum_df['total_sum']

    def format_output(row, mode='train'):
        """根據 Train 或 Test 格式化輸出字串"""
        count = row[f'{mode}_sum']
        ratio = row[f'{mode}_ratio']
        total = row['total_sum']
        
        # 使用 f-string 進行格式化，保留較多小數位數以符合您的範例
        return f"{int(count)} ({ratio} = {int(count)} / {int(total)} )"

    print("--- 訓練集 (Train) 每個類別的總推文數 ---")
    # 對 train 欄位應用格式化函數
    train_formatted_output = combined_sum_df.apply(lambda row: format_output(row, mode='train'), axis=1)
    print(train_formatted_output.to_string(header=False))

    print("\n--- 測試集 (Test) 每個類別的總推文數 ---")
    # 對 test 欄位應用格式化函數
    test_formatted_output = combined_sum_df.apply(lambda row: format_output(row, mode='test'), axis=1)
    print(test_formatted_output.to_string(header=False))

    return train_df, test_df



# --- 用日期為單位把資料切成 8:2 (train : test) ---
def splitset_dates(COIN_SHORT_NAME):

    # --- 讀取每條推文的 ID .pkl ---
    with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids{SUFFIX}.pkl", "rb") as f:   # rb = read binary
        ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)]
    dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array

    # --- 讀取每條推文的 價格變化率 ---
    price_diff_rate = np.load(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_price_diff_original{SUFFIX}.npy")
    price_diff_categories = categorize_array_multi(price_diff_rate)
    
    # 如果是 bytes，要轉成 str
    if isinstance(dates[0], bytes):
        dates = dates.astype(str)

    # 轉成 datetime 方便排序
    dates_dt = pd.to_datetime(dates, format="%Y-%m-%d")

    # 統計每天出現次數
    unique_dates, counts = np.unique(dates_dt, return_counts=True)

    if len(price_diff_rate) != len(unique_dates):
        print("⚠️ 警告：價格變化率陣列長度與唯一日期數量不匹配。請檢查 price_diff_rate 的生成邏輯。")
        print(f" price_diff_rate 長度: {len(price_diff_rate)}")
        print(f" unique_dates 長度: {len(unique_dates)}")
        raise ValueError(f"價格變化率陣列長度與唯一日期數量不匹配")

    # 建立 DataFrame
    df = pd.DataFrame({
        "date": unique_dates,
        "tweet_count": counts,
        "price_diff_rate": price_diff_rate,
        "category": price_diff_categories
    })

    # 加上 weekday
    df["weekday_name"] = df["date"].dt.day_name()   # e.g., Friday
    df["weekday_num"] = df["date"].dt.weekday       # Monday=0, Sunday=6

    # 儲存為 CSV
    csv_output_path = f"{INPUT_PATH}/coin_price"
    os.makedirs(csv_output_path, exist_ok=True)
    df.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_filtered_tweet_count{SUFFIX}.csv", index=False, encoding="utf-8-sig")

    print(f"✅ 已儲存 CSV 至: {csv_output_path}/{COIN_SHORT_NAME}_filtered_tweet_count{SUFFIX}.csv")
    # input("pause...")


    # --- 切分 ---
    train_df, test_df = split_by_week(df)
    print("✅ 已利用週為單位切割好資料集，準備進行微調...")
    # input("pause...")

    # --- 隨機交換微調 tolerance: 誤差推文數值 (100 => +-100) ---
    all_categories = train_df['category'].unique()  # 找出所有類別

    # 迭代調整每個類別
    for cat_id in sorted(all_categories):
        print(f"\n======== 開始調整類別 {cat_id} ========")
        train_df, test_df = balance_category_proportion_swap(train_df, test_df, cat_id, target_ratio=0.8, tolerance=0.001)

    print("✅ 已成功完成微調")
    # input("pause...")


    # 先按照 tweet_count 由大到小排序
    train_df_sorted = train_df.sort_values(by='tweet_count', ascending=False)
    test_df_sorted  = test_df.sort_values(by='tweet_count', ascending=False)

    csv_output_path = f"{INPUT_PATH}/split_dates"
    os.makedirs(csv_output_path, exist_ok=True)
    train_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_train_dates{SUFFIX}.csv", index=False, encoding="utf-8-sig")
    test_df_sorted.to_csv(f"{csv_output_path}/{COIN_SHORT_NAME}_test_dates{SUFFIX}.csv", index=False, encoding="utf-8-sig")
    print("✅ 已成功儲存資料集的日期")
    input("pause...")

    # --- 展開成每條推文一個單位 ---
    dates_train_expanded = expand_by_tweet(train_df)
    dates_test_expanded = expand_by_tweet(test_df)

    return dates_train_expanded, dates_test_expanded


# --- 用平衡好的結果來按日期切割 X, Y，並可選擇是否要再分出 val (預設 False) ---
def splitset_XY(COIN_SHORT_NAME):

    # 讀取稀疏矩陣
    X = sparse.load_npz(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_X_sparse{SUFFIX}.npz")  # 二維陣列：colunm(關鍵詞) row(某天某推文) (但這裡是稀疏矩陣的格式)
    
    # 一維陣列：存放與 X row 對應的 ID
    with open(f"{INPUT_PATH}/keyword/{COIN_SHORT_NAME}_ids{SUFFIX}.pkl", "rb") as f:   # rb = read binary
        ids = pickle.load(f)  # array[('coin', 'date', 'no.'), (str, '%Y-%m-%d', int)
    ids = np.array(ids)  # 把 ids 轉成 numpy array
    dates = np.array([row[1] for row in ids])  # 只把 'date' 取出來，並轉成 np.array

    # 讀取三個集合的日期 (切割好的 CSV) 
    train_dates = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_train_dates{SUFFIX}.csv")['date']
    test_dates = pd.read_csv(f"{INPUT_PATH}/split_dates/{COIN_SHORT_NAME}_test_dates{SUFFIX}.csv")['date']

    # 讀取 price_diff.npy
    Y_all = np.load(f"{INPUT_PATH}/coin_price/{COIN_SHORT_NAME}_price_diff{SUFFIX}.npy")  # shape = (總推文數, 1)


    # 輸出長度 確保一致性
    print(f"{COIN_SHORT_NAME}{SUFFIX}：")
    print("X.shape[0] =", X.shape[0])
    print("ids.shape[0] =", len(ids))
    print("Y.shape[0] =", Y_all.shape[0],"\n")

    print(f"{COIN_SHORT_NAME}{SUFFIX} 的 Y_all 有 {np.sum(Y_all == 0)} 個 0")



    # 把 date 轉成 datetime 格式，方便比對
    dates_datetime = pd.to_datetime(dates)

    print(dates_datetime)

    # 轉成 datetime
    train_dates = pd.to_datetime(train_dates)
    test_dates  = pd.to_datetime(test_dates)

    print(train_dates)

    # 找出 index   逐筆檢查 date 中的每一個值，判斷它是否在 train_dates 裡
    train_mask = dates_datetime.isin(train_dates)
    test_mask  = dates_datetime.isin(test_dates)

    # 切割 X
    X_train = X[train_mask, :]
    X_test  = X[test_mask, :]

    # 切割 Y
    Y_train = Y_all[train_mask]  # shape = (len(train_mask), 1)
    Y_test  = Y_all[test_mask]

    # 切割 ids
    ids_train = ids[train_mask]
    ids_test  = ids[test_mask]

    print(f"{COIN_SHORT_NAME}{SUFFIX} 的 Y_train 有 {np.sum(Y_train == 0)} 個 0")
    print(f"{COIN_SHORT_NAME}{SUFFIX} 的 Y_test 有 {np.sum(Y_test == 0)} 個 0")

    # 列印各個幣種 split 後 每個資料集的漲跌比例
    # print(Y_train.shape)
    # print(Y_test.shape)
    print(ids_train.shape)
    print(ids_test.shape)
    # for i in range(LABEL_COUNT):
    #     print(f"第 {i} 組 Ｙ")
    print_label_distribution(Y_train, Y_test, COIN_SHORT_NAME)

    return X_train, X_test, Y_train, Y_test, ids_train, ids_test



# --- 將不必要的關鍵詞與推文刪除 ---
def filter_XY(X_train, X_test, Y_train, Y_test, ids_train, ids_test, all_vocab):
    '''重複 功能 1, 功能 2, 功能 3 直到沒有東西可以刪為止'''

    # 定義 功能 1
    def function_1(X, Y, ids):
        row_sums = np.array(X.sum(axis=1)).ravel()
        valid_rows = np.where(row_sums > 0)[0]
        invalid_rows = X.shape[0] - len(valid_rows)

        X = X[valid_rows, :]  # 也可寫 X = X[valid_rows]
        Y = Y[valid_rows]
        ids = ids[valid_rows]
        
        return X, Y, ids, invalid_rows, len(valid_rows)
            

    invalid_rows = -1
    delete_only_test = -1

    total_delete_rows = 0
    total_delete_columns = 0

    # 若還有功能是可以刪資料的，就再繼續跑
    while invalid_rows != 0 or delete_only_test != 0:

        # --- 功能 1: 刪掉沒有任何關鍵詞的推文 (刪 row) ---
        # train
        X_train, Y_train, ids_train, train_invalid_rows, train_valid_rows = function_1(X_train, Y_train, ids_train)
        
        # test
        X_test, Y_test, ids_test, test_invalid_rows, test_valid_rows = function_1(X_test, Y_test, ids_test)

        # 計算保留、刪掉的筆數
        valid_rows = train_valid_rows + test_valid_rows
        invalid_rows = train_invalid_rows + test_invalid_rows
            
        # 只看 Train 的數量
        total_delete_rows += train_invalid_rows
        print("功能 1: 刪掉沒有任何關鍵詞的推文 (row):")
        print(f"\tTrain 保留 row 數量: {train_valid_rows}")
        print(f"\tTrain 刪掉 row 數量: {train_invalid_rows}\n")


        # --- 功能 2: 只保留 train 出現過的關鍵詞 (刪 column) --- 
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

        # 每個關鍵詞的出現次數統計
        col_sums = np.array(X_train.sum(axis=0)).ravel()
        keyword_counts = {all_vocab[i]: int(col_sums[i]) for i in range(len(all_vocab))}
        stats_output_path = os.path.join(INPUT_PATH, "keyword", f"keyword_counts{SUFFIX}.json")
        with open(stats_output_path, "w", encoding="utf-8") as f:
            json.dump(keyword_counts, f, ensure_ascii=False, indent=4)

        # column 過濾
        X_train = X_train[:, keep_cols]  # [:, keep_cols] 表示「保留所有 row，但只取出 keep_cols 這些 column」。
        X_test  = X_test[:, keep_cols]
        if all_vocab is not None:
            filtered_vocab = [all_vocab[i] for i in keep_cols]

        delete_only_test = orig_cols - new_cols
        total_delete_columns += delete_only_test
        print("功能 2: 只保留 train 出現過的關鍵詞 (column):")
        print(f"\t原始 column 數量: {orig_cols}")
        print(f"\t保留 column 數量: {new_cols}")
        print(f"\t刪掉 column 數量: {delete_only_test}\n")

        with open(os.path.join(f"{INPUT_PATH}/keyword", f"filtered_keywords{SUFFIX}.json"), "w", encoding="utf-8") as f:
            json.dump(filtered_vocab, f, ensure_ascii=False, indent=4)  

    print(f"總共刪除 {total_delete_rows} 個推文 (row), {total_delete_columns} 個關鍵詞 (column)")
    print(f"Train 總共保留 {X_train.shape[0]} 個推文 (row), {X_train.shape[1]} 個關鍵詞 (column)\n")
    print(f"已輸出所有關鍵詞出現次數統計到 {stats_output_path}")
    print(f"已輸出所有被過濾的關鍵詞到 {INPUT_PATH}/keyword\n")

    return X_train, X_test, Y_train, Y_test, ids_train, ids_test



# --- 打亂順序 (shuffle) ---
def shuffle_XY(X, Y, ids, seed=42):
    """
    Shuffle X and Y in unison.
    X: np.ndarray 或 scipy.sparse 矩陣
    Y: np.ndarray 一維標籤
    seed: 隨機種子
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(Y.shape[0])  # 取得樣本數 indices = [0, 1, 2, ... , len(X)-1]
    rng.shuffle(indices)  # 把 indices 隨機重新排序

    X_shuffled = X[indices, :]  # 按照 indices 的順序重新排列
    Y_shuffled = Y[indices]
    ids_shuffled = ids[indices]

    return X_shuffled, Y_shuffled, ids_shuffled



# --- 計算每個日期總共的數量 (用以確認是否有切正確) ---
def count_per_day(ids, dataset_name):
    """
    dates: array-like, 每條推文的日期 (str 或 np.datetime64)
    dataset_name: 用於打印
    """
    dates = [row[1] for row in ids]
    # 如果是 bytes，先轉成 str
    if isinstance(dates[0], bytes):
        dates = dates.astype(str)

    # 轉成 datetime
    dates_dt = pd.to_datetime(dates)

    # 計算每天出現次數
    date_counts = dates_dt.value_counts().sort_index()  # 按日期排序
    df_counts = pd.DataFrame({"date": date_counts.index, "tweet_count": date_counts.values})

    df_counts.to_csv(f"{INPUT_PATH}/dates_{dataset_name}_counts{SUFFIX}.csv", index=False)

    return df_counts



# --- 將三種幣種的 X, Y 合併成完整的模型輸入值 (輸出 .npy 檔) ---
def merge(DOGE_X_train, DOGE_X_test, DOGE_Y_train, DOGE_Y_test,
          PEPE_X_train, PEPE_X_test, PEPE_Y_train, PEPE_Y_test,
          TRUMP_X_train, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_test,
          DOGE_ids_train, DOGE_ids_test,
          PEPE_ids_train, PEPE_ids_test,
          TRUMP_ids_train, TRUMP_ids_test,
          all_vocab):

    # 合併 X（稀疏矩陣用 sparse.vstack）
    X_train_list = [DOGE_X_train, PEPE_X_train, TRUMP_X_train]
    X_test_list  = [DOGE_X_test, PEPE_X_test, TRUMP_X_test]

    X_train = sparse.vstack(X_train_list, format="csr")  # np.vstack = vertical stack，把多個矩陣在「列方向」堆疊起來
    X_test  = sparse.vstack(X_test_list, format="csr")

    # 合併 Y
    Y_train = np.concatenate([DOGE_Y_train, PEPE_Y_train, TRUMP_Y_train])  # np.concatenate = 把多個一維陣列串接起來
    Y_test  = np.concatenate([DOGE_Y_test, PEPE_Y_test, TRUMP_Y_test])

    # 合併 ids
    ids_train = np.concatenate([DOGE_ids_train, PEPE_ids_train, TRUMP_ids_train])
    ids_test  = np.concatenate([DOGE_ids_test, PEPE_ids_test, TRUMP_ids_test])


    # 將不必要的關鍵詞與推文刪除
    X_train, X_test, Y_train, Y_test, ids_train, ids_test = filter_XY(X_train, X_test, Y_train, Y_test, ids_train, ids_test, all_vocab)
    # print(ids_train.shape)


    # 打亂順序
    X_train, Y_train, ids_train = shuffle_XY(X_train, Y_train, ids_train)
    X_test,  Y_test,  ids_test  = shuffle_XY(X_test, Y_test, ids_test)

    # 檢查資料集維度
    assert X_train.shape[0] == Y_train.shape[0] == len(ids_train), "Train 維度不一致!"
    assert X_test.shape[0] == Y_test.shape[0] == len(ids_test), "Test 維度不一致!"

    # 列印 merge, filter 後 每個資料集的漲跌比例
    # print(Y_train.shape)
    # for i in range(LABEL_COUNT):
    #     print(f"第 {i} 組 Ｙ")
    print_label_distribution(Y_train, Y_test, "ALL")

    print(f"Y_train{SUFFIX} 有 {np.sum(Y_train == 0)} 個 0")
    print(f"Y_test{SUFFIX} 有 {np.sum(Y_test == 0)} 個 0")

    # 將 Y 都變成 類別
    Y_train = categorize_array_multi(Y_train)
    Y_test = categorize_array_multi(Y_test)

    # 儲存
    sparse.save_npz(f"{INPUT_PATH}/X_train{SUFFIX}.npz", X_train)
    sparse.save_npz(f"{INPUT_PATH}/X_test{SUFFIX}.npz", X_test)
    np.savez_compressed(f"{INPUT_PATH}/Y_train{SUFFIX}.npz", Y=Y_train)
    np.savez_compressed(f"{INPUT_PATH}/Y_test{SUFFIX}.npz",  Y=Y_test)


    if ids_train is not None:
        print(ids_train.shape)
        print(ids_test.shape)
        with open(f"{INPUT_PATH}/ids_train{SUFFIX}.pkl", 'wb') as file:
            pickle.dump(ids_train.tolist(), file)
        with open(f"{INPUT_PATH}/ids_test{SUFFIX}.pkl", 'wb') as file:
            pickle.dump(ids_test.tolist(), file)




    print(f"Merge{SUFFIX} 完成，資料已輸出到 ../data/ml/dataset\n")

    # 計算每個資料集中每天的推文總數
    count_per_day(ids_train, "train")
    count_per_day(ids_test, "test")

    print("已將不同資料集每天的推文總數輸出為 csv 到 ../data/ml/dataset\n")





# --- 列印平衡好的結果 ---
def print_split_number(train_expanded, test_expanded, COIN_SHORT_NAME):
    sum = len(train_expanded) + len(test_expanded)
    print(f"{COIN_SHORT_NAME}{SUFFIX}：")
    print(f"理論值 Train: {int(sum * 0.8)},                Test: {int(sum * 0.1)}")
    print(f"實際值 Train: {len(train_expanded)} ({round((len(train_expanded) / sum), 10)}), Test: {len(test_expanded)} ({round((len(test_expanded) / sum), 10)})\n")



def main():

    # 分別切資料集
    DOGE_dates_train_expanded, DOGE_dates_test_expanded = splitset_dates("DOGE")
    print_split_number(DOGE_dates_train_expanded, DOGE_dates_test_expanded, "DOGE")
    DOGE_X_train, DOGE_X_test, DOGE_Y_train, DOGE_Y_test, DOGE_ids_train, DOGE_ids_test = splitset_XY("DOGE")  # 若要分出 val => splitset_XY("DOGE", True)
    
    PEPE_dates_train_expanded, PEPE_dates_test_expanded = splitset_dates("PEPE")
    print_split_number(PEPE_dates_train_expanded, PEPE_dates_test_expanded, "PEPE")
    PEPE_X_train, PEPE_X_test, PEPE_Y_val, PEPE_Y_test, PEPE_ids_train, PEPE_ids_test = splitset_XY("PEPE")
    
    TRUMP_dates_train_expanded, TRUMP_dates_test_expanded = splitset_dates("TRUMP")
    print_split_number(TRUMP_dates_train_expanded, TRUMP_dates_test_expanded, "TRUMP")
    TRUMP_X_train, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_test, TRUMP_ids_train, TRUMP_ids_test = splitset_XY("TRUMP")
    
    
    # 讀取所有關鍵詞的名字
    json_path = os.path.join("../data/keyword/machine_learning", f"all_keywords{SUFFIX}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    all_vocab = list(vocab)

    print(DOGE_ids_train.shape)
    print(DOGE_ids_test.shape)
    print(PEPE_ids_train.shape)
    print(PEPE_ids_test.shape)
    print(TRUMP_ids_train.shape)
    print(TRUMP_ids_test.shape)

    # 合併資料集
    merge(DOGE_X_train, DOGE_X_test, DOGE_Y_train, DOGE_Y_test,
        PEPE_X_train, PEPE_X_test, PEPE_Y_val, PEPE_Y_test,
        TRUMP_X_train, TRUMP_X_test, TRUMP_Y_train, TRUMP_Y_test,
        DOGE_ids_train, DOGE_ids_test,
        PEPE_ids_train, PEPE_ids_test,
        TRUMP_ids_train, TRUMP_ids_test,
        all_vocab)
    

if __name__ == "__main__":
    main()