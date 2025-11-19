from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

'''可修改參數'''
COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]

MODEL_NAME = ["random_forest", "rf"]  # ["logistic_regression", "logreg"] ["random_forest", "rf"] ["SGD", "sgd"]    

INPUT_PATH = "../data/ml/dataset"

INPUT_FIRST_CLASSIFIER_PATH = f"../data/ml/classification/{MODEL_NAME[0]}"

OUTPUT_PATH = "../data/ml/dataset/final_input/price_classifier"

MERGE_CLASSIFIER_1_RESULT = True  # 看是否要合併第一個分類器的預測結果

IS_FILTERED = True  # 看是否有分 normal 與 bot

IS_RUN_AUGUST = False  # 看現在是不是要跑 2025/08 的資料(未完成)

IS_CATEGORY_Y = True  # 看是否要先把 Y 變成類別 (0 ~ 4)
'''可修改參數'''

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
SUFFIX_AUGUST   = "_202508" if IS_RUN_AUGUST else ""
SUFFIX_CLASSIFIER_1 = "" if MERGE_CLASSIFIER_1_RESULT else "_non_classifier_1"




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




def merge():
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    ids_train = []
    ids_test = []
    all_coin_dates = set()  # 用集合自動去重
    ids_all_coin = []

    # 若是要跑 8月 的資料
    X_single_coin_dict = {}
    # Y_single_coin_dict = {}
    ids_single_coin_dict = {}

    # 將不同幣種的 X, Y 分別讀取進來
    for coin_short_name in COIN_SHORT_NAME:
        print(f"\n🚩 正在處理 {coin_short_name} ...")

        # --- 讀取 X ---
        X_diff_past = np.load(f"{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_price_diff_past5days{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")  # 讀取 前面幾天 的 價差、價錢變化率
        X_XGBoost = np.load(f"{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_XGBoost_features.npy")  # 讀取 XBGoost 所使用的 features
        X_first_classifier = np.load(f"{INPUT_FIRST_CLASSIFIER_PATH}/keyword_classifier/single_coin_result/{coin_short_name}/{coin_short_name}_{MODEL_NAME[1]}_classifier_1_result{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")  # 讀取 第一個分類器 預測的結果
        X_sentiment_1 = np.load(f"{INPUT_PATH}/X_input/price_classifier/sentiments/bert_43/{coin_short_name}_bert_43.npy")  # 讀取以天為單位情緒分析的結果 (model: bert-43)
        # X_sentiment_2 = np.load(f"{INPUT_PATH}/X_input/price_classifier/sentiments/")  # 讀取以天為單位情緒分析的結果 (model: cardiffnlp)

        print("X_sentiment_1.shape:", X_sentiment_1.shape)


        # --- 讀取 X 的日期參考資料 ---
        XGBoost_dates = np.loadtxt(f"{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_XGBoost_dates.txt", dtype=str)  # 讀取 XBGoost 所使用的 dates
        with open(f"{INPUT_PATH}/final_input/keyword_classifier/ids_train{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", "rb") as f:   # 讀取一開始訓練用的 ids
            ids_train_classifier_1 = pickle.load(f)
            print("len(ids_train_classifier_1):", len(ids_train_classifier_1))
        with open(f"{INPUT_PATH}/final_input/keyword_classifier/ids_test{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", "rb") as f:   # 讀取一開始訓練用的 ids
            ids_test_classifier_1 = pickle.load(f)
            print("len(ids_test_classifier_1):", len(ids_test_classifier_1))

        ids = ids_train_classifier_1 + ids_test_classifier_1
        print("len(ids):", len(ids))
        ids = [(c, d, no) for (c, d, no) in ids if c == coin_short_name]
        print("len(ids):", len(ids))
        print("ids[:10]:\n", ids[:10])

        
        all_coin_dates.update([(c, d) for (c, d, _) in ids])  # 只取 (coin, date) 加入集合

        # 先把 all_coin_dates 只保留當前幣種的日期
        current_coin_dates = set([d for (c, d) in all_coin_dates if c == coin_short_name])

        # print("X_XGBoost.shape", X_XGBoost.shape)
        # print("XGBoost_dates[-10:]", XGBoost_dates[-10:])
        # print("current_coin_dates：", sorted(current_coin_dates))
        # print("len(current_coin_dates)：", len(current_coin_dates))


        # --- 讀取 現在幣種 的 merge 過濾前 日期 ---
        with open(f"{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_current_tweet_count{SUFFIX_FILTERED}{SUFFIX_AUGUST}.json", "r", encoding="utf-8") as f:
            single_coin_original_dates = json.load(f)
        single_coin_original_dates = np.array(list(single_coin_original_dates.keys()))  # 取出所有日期（也就是 key）
        single_coin_original_dates = np.array([
            datetime.strptime(d, "%Y/%m/%d").strftime("%Y-%m-%d")
            for d in single_coin_original_dates
        ])  # 把 "%Y/%m/%d" 轉成 "%Y-%m-%d"
        print("len(single_coin_original_dates):", len(single_coin_original_dates))
        print("single_coin_original_dates[:10]:\n", single_coin_original_dates[:10])
        

        # 建立 mask，只保留在 all_coin_dates 裡的日期
        mask_XGBoost = [d in current_coin_dates for d in XGBoost_dates]
        print("len(mask):", len(mask_XGBoost))

        # 用 mask 過濾 X_XGBoost 與 XGBoost_dates
        X_XGBoost = X_XGBoost[mask_XGBoost]
        XGBoost_dates = XGBoost_dates[mask_XGBoost]  # 為了看 X_XGBoost 有沒有刪正確
        print("XGBoost_dates.shape:", XGBoost_dates.shape)
        print("XGBoost_dates[:10]:\n", XGBoost_dates[:10])

        # 用 mask 過濾 X_sentiment
        print("set(XGBoost_dates) - set(single_coin_original_dates):", set(XGBoost_dates) - set(single_coin_original_dates))
        mask_sentiment = [d in XGBoost_dates for d in single_coin_original_dates]
        X_sentiment_1 = X_sentiment_1[mask_sentiment]
        # X_sentiment_2 = X_sentiment_2[mask]
        single_coin_original_dates = single_coin_original_dates[mask_sentiment]  # 為了看 X_sentiment 有沒有刪正確
        print("X_sentiment_1.shape:", X_sentiment_1.shape)
        print("single_coin_original_dates.shape:", single_coin_original_dates.shape)
        print("single_coin_original_dates[:10]:\n", single_coin_original_dates[:10])

        current_coin_ids = set([(c, d) for (c, d) in all_coin_dates if c == coin_short_name])
        # ids_all_coin += sorted(current_coin_ids)
        print(f"去掉重複日期後 {coin_short_name} 的 (coin, date) 數量: {len(current_coin_ids)}\n")
        print(f"{coin_short_name} 的 XGBoost 相關特徵的日期：(用來檢查 X_XGBoost 有沒有取正確 DOGE、TRUMP前面會少 13 天)\n{XGBoost_dates[:10]}\n")

        # print("X_diff_past.shape:", X_diff_past.shape)
        # print("X_XGBoost.shape:", X_XGBoost.shape)
        # print("X_first_classifier.shape:", X_first_classifier.shape)

        # --- 讀取 Y ---
        Y_single_coin = np.load(f"{INPUT_PATH}/y_input/{coin_short_name}/{coin_short_name}_price_diff_original{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy")  # 讀取 明天 的價錢變化率 (price_diff_rate_tomorrow)
        print("Y_single_coin.shape:", Y_single_coin.shape)

        # # --- 對齊時間軸 ---
        # if coin_short_name == "PEPE":  # 因為 PEPE 不是從發售就開始抓推文 所以 X_XGBoost 沒有被跳過 13 天
        #     start_idx = 5   # 因為 X_diff_past 特徵要跳過前 5 天
        #     X_XGBoost = X_XGBoost[start_idx:]
        #     X_first_classifier = X_first_classifier[start_idx:]
        #     Y_single_coin = Y_single_coin[start_idx:]
        # elif coin_short_name == "TRUMP":  # 因為 TRUMP 在 2025-01-27 沒有抓到資料 但此日期包含在 X_XGBoost 所跳過的 13 天內
        #     start_idx = 12   # 因為 XGBoost 特徵要跳過前 13 天 但要 -1
        #     X_diff_past = X_diff_past[(start_idx - 5):]  # 原本少 5 天 → 再切掉到 7
        #     X_first_classifier = X_first_classifier[start_idx:]
        #     Y_single_coin = Y_single_coin[start_idx:]
        # elif coin_short_name == "DOGE":
        #     start_idx = 13   # 因為 XGBoost 特徵要跳過前 13 天
        #     X_diff_past = X_diff_past[(start_idx - 5):]  # 原本少 5 天 → 再切掉到 8
        #     X_first_classifier = X_first_classifier[start_idx:]
        #     Y_single_coin = Y_single_coin[start_idx:]

        # --- 對齊時間軸（從後面對齊） ---
        min_len = min(len(X_diff_past), len(X_XGBoost), len(X_first_classifier), len(Y_single_coin))
        print("len(X_diff_past), len(X_XGBoost), len(X_first_classifier), len(Y_single_coin):",len(X_diff_past), len(X_XGBoost), len(X_first_classifier), len(Y_single_coin))
        X_diff_past = X_diff_past[-min_len:]
        X_XGBoost = X_XGBoost[-min_len:]
        X_sentiment_1 = X_sentiment_1[-min_len:]
        # X_sentiment_2 = X_sentiment_2[-min_len:]
        X_first_classifier = X_first_classifier[-min_len:]
        Y_single_coin = Y_single_coin[-min_len:]
        single_coin_ids = (sorted(current_coin_ids)[-min_len:])
        # ids_all_coin += (sorted(current_coin_ids)[-min_len:])

        print(f"目前 single_coin_ids (要輸出的 ids) 內容：\n{single_coin_ids[:10]}\n")
        print(f"single_coin_ids (要輸出的 ids) 的長度：{len(single_coin_ids)}\n")


        # --- 合併特徵 ---
        if MERGE_CLASSIFIER_1_RESULT:
            X_single_coin = np.hstack([X_diff_past, X_XGBoost, X_first_classifier.reshape(-1, 1), X_sentiment_1])  # , X_sentiment_2
        else:
            X_single_coin = np.hstack([X_diff_past, X_XGBoost, X_sentiment_1])  # , X_sentiment_2

        X_single_coin_dict[coin_short_name] = X_single_coin
        # Y_single_coin_dict[coin_short_name] = Y_single_coin
        ids_single_coin_dict[coin_short_name] = single_coin_ids  # sorted(current_coin_ids)[-min_len:]
        

        # --- 依照第一個分類器所切割的資料集來分 ---
        # 讀取每個幣種第一個分類的資料集日期
        single_coin_train_date = pd.read_csv(f"../data/ml/dataset/split_dates/{coin_short_name}_train_dates{SUFFIX_FILTERED}.csv")
        single_coin_test_date = pd.read_csv(f"../data/ml/dataset/split_dates/{coin_short_name}_test_dates{SUFFIX_FILTERED}.csv")
        # single_coin_val_date_only = pd.read_csv(f"../data/ml/dataset/split_dates/{coin_short_name}_val_dates{SUFFIX_FILTERED}.csv")
        # single_coin_test_date = pd.concat([single_coin_test_date_only, single_coin_val_date_only], ignore_index=True)  # 將 test val 合併

        single_coin_train_date = set(single_coin_train_date["date"])
        single_coin_test_date = set(single_coin_test_date["date"])

        # 建立對應 train/test 的 mask（布林列表）
        train_mask = [d in single_coin_train_date for (c, d) in single_coin_ids]
        test_mask = [d in single_coin_test_date for (c, d) in single_coin_ids]

        # 使用 mask 對 y_true, y_pred, y_dates 分割
        single_coin_X_train_set = [Xsc for Xsc, m in zip(X_single_coin, train_mask) if m]
        single_coin_y_train_set = [Ysc for Ysc, m in zip(Y_single_coin, train_mask) if m]
        single_coin_ids_train_set = [ids for ids, m in zip(single_coin_ids, train_mask) if m]
        print(f"{coin_short_name} single_coin_ids_train_set[:10]:\n", single_coin_ids_train_set[:10])

        single_coin_X_test_set = [Xsc for Xsc, m in zip(X_single_coin, test_mask) if m]
        single_coin_y_test_set = [Ysc for Ysc, m in zip(Y_single_coin, test_mask) if m]
        single_coin_ids_test_set = [ids for ids, m in zip(single_coin_ids, test_mask) if m]
        print(f"{coin_short_name} single_coin_ids_test_set[:10]:\n", single_coin_ids_test_set[:10])
        input("按 Enter 以繼續 ...")


        # --- 存進總集合 ---
        X_train.append(single_coin_X_train_set)
        X_test.append(single_coin_X_test_set)
        Y_train.append(single_coin_y_train_set)
        Y_test.append(single_coin_y_test_set)
        ids_train.append(single_coin_ids_train_set)
        ids_test.append(single_coin_ids_test_set)


    if not IS_RUN_AUGUST:
        # --- 把三個幣種合併成一個大陣列 ---
        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)
        Y_train = np.concatenate(Y_train)
        Y_test = np.concatenate(Y_test)
        ids_train = np.vstack(ids_train)
        ids_test = np.vstack(ids_test)


        X_doge = None
        X_pepe = None
        X_trump = None
        ids_doge = None
        ids_pepe = None
        ids_trump = None

        print("\n✅ 已經完成合併\n")

        return X_train, X_test, Y_train, Y_test, ids_train, ids_test, X_doge, X_pepe, X_trump, ids_doge, ids_pepe, ids_trump
    
    else:
        X_train = None
        X_test = None
        Y_train = None
        Y_test = None
        ids_train = None
        ids_test = None

        X_doge = X_single_coin_dict["DOGE"]
        X_pepe = X_single_coin_dict["PEPE"]
        X_trump = X_single_coin_dict["TRUMP"]
        ids_doge = ids_single_coin_dict["DOGE"]
        ids_pepe = ids_single_coin_dict["PEPE"]
        ids_trump = ids_single_coin_dict["TRUMP"]

        return X_train, X_test, Y_train, Y_test, ids_train, ids_test, X_doge, X_pepe, X_trump, ids_doge, ids_pepe, ids_trump
    



def export_to_csv(X, Y, ids, output_path):
    # 把 ids 拆成 coin / date
    coins = [c for c, d in ids]
    dates = [d for c, d in ids]

    # 建立 DataFrame
    df = pd.DataFrame({
        "coin": coins,
        "date": dates,
        "label": Y
    })

    # 如果 X 有 feature，則展開
    if X.ndim == 2:
        feature_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df = pd.concat([df, feature_df], axis=1)
    else:
        # 一維或其他情況直接存
        df["feature"] = X

    # 存成 CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ 輸出完成: {output_path}")



def main():
    if not IS_RUN_AUGUST:
        print("目前沒有跑 august")
        X_train, X_test, Y_train, Y_test, ids_train, ids_test, _, _, _, _, _, _ = merge()

        print("X_train.shape =", X_train.shape)
        print("Y_train.shape =", Y_train.shape)
        print("len(ids_train) =", len(ids_train))

        print("X_test.shape =", X_test.shape)
        print("Y_test.shape =", Y_test.shape)
        print("len(ids_test) =", len(ids_test))

        if IS_CATEGORY_Y:
            Y_train = categorize_array_multi(Y_train)
            Y_test = categorize_array_multi(Y_test)


        # 輸出 merge 好的資料到 csv 看，用來檢查是否有問題
        export_to_csv(X_train, Y_train, ids_train, f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_train_merged_dataset{SUFFIX_FILTERED}{SUFFIX_AUGUST}{SUFFIX_CLASSIFIER_1}.csv")
        export_to_csv(X_test, Y_test, ids_test, f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_test_merged_dataset{SUFFIX_FILTERED}{SUFFIX_AUGUST}{SUFFIX_CLASSIFIER_1}.csv")

        

        # print("🚩 打亂前：")
        # print("\nX_train 預覽：\n", X_train[:10])
        # print("\nY_train 預覽：\n", Y_train[:10])
        # print("\ids_train 預覽：\n", ids_train[:10])
        # print("\nX_test 預覽：\n", X_test[:10])
        # print("\nY_test 預覽：\n", Y_test[:10])
        # print("\ids_test 預覽：\n", ids_test[:10])

        # --- 打亂 X, Y, ids ---
        X_train, Y_train, ids_train = shuffle_XY(X_train, Y_train, ids_train)
        X_test, Y_test, ids_test = shuffle_XY(X_test, Y_test, ids_test)

        # print("\n🚩 打亂後：")
        # print("\nX_train 預覽：\n", X_train[:10])
        # print("\nY_train 預覽：\n", Y_train[:10])
        # print("\ids_train 預覽：\n", ids_train[:10])
        # print("\nX_test 預覽：\n", X_test[:10])
        # print("\nY_test 預覽：\n", Y_test[:10])
        # print("\ids_test 預覽：\n", ids_test[:10])

        # 儲存
        np.save(f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_X_train_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}{SUFFIX_CLASSIFIER_1}.npy", X_train)
        np.save(f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_Y_train_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy", Y_train)
        with open(f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_ids_train_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'wb') as file:
            pickle.dump(ids_train, file)  # 這裡只會存 ('coin', 'date') 且每個日期只有一筆

        np.save(f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_X_test_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}{SUFFIX_CLASSIFIER_1}.npy", X_test)
        np.save(f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_Y_test_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.npy", Y_test)
        with open(f"{OUTPUT_PATH}/{MODEL_NAME[0]}/{MODEL_NAME[1]}_ids_test_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", 'wb') as file:
            pickle.dump(ids_test, file)  # 這裡只會存 ('coin', 'date') 且每個日期只有一筆

        print(f"\n✅ 已成功儲存至 {OUTPUT_PATH}/{MODEL_NAME[0]}/\n")




        # Y = categorize_array_multi(Y)
        # print("Y[:10]:", Y[:30])

        # y_pred = []
        # for csn, delete in zip(COIN_SHORT_NAME, [13, 0, 12]):
        #     print(f"目前正在執行 {csn} ...\n")
        #     Y_PRED_PATH = Path(f'../data/ml/classification/{"logistic_regression"}/{csn}_{MODEL_NAME}_classifier_1_result.npy')
            
        #     y_pred += (np.load(Y_PRED_PATH).tolist())[delete:]
        # y_pred = np.array(y_pred)[indices]
        # print("y_pred[:10]:", y_pred[:30])

        # y_true_train, y_true_test, y_pred_train, y_pred_test = train_test_split(
        #     Y, y_pred, test_size=0.2, random_state=42, shuffle=True
        # )


        # print()
        # print(classification_report(y_true_train, y_pred_train, digits=3, target_names=['大跌', '小跌', '持平', '小漲', '大漲']))
        # print()
        # print(classification_report(y_true_test, y_pred_test, digits=3, target_names=['大跌', '小跌', '持平', '小漲', '大漲']))

    else:
        print("目前正在跑 august")
        _, _, _, X_doge, X_pepe, X_trump, ids_doge, ids_pepe, ids_trump = merge()

        print("X_doge.shape =", X_doge.shape)
        print("X_pepe.shape =", X_pepe.shape)
        print("X_trump.shape =", X_trump.shape)
        print("len(ids_doge) =", len(ids_doge))
        print("len(ids_pepe) =", len(ids_pepe))
        print("len(ids_trump) =", len(ids_trump))

        print("🚩 預覽：")
        print("\nX_doge 預覽：\n", X_doge[:10])
        print("\nX_pepe 預覽：\n", X_pepe[:10])
        print("\nX_trump 預覽：\n", X_trump[:10])
        print("\nids_doge 預覽：\n", ids_doge[:10])
        print("\nids_pepe 預覽：\n", ids_pepe[:10])
        print("\nids_trump 預覽：\n", ids_trump[:10])

        # 儲存
        X_list = [X_doge, X_pepe, X_trump]
        ids_list = [ids_doge, ids_pepe, ids_trump]

        for coin_short_name, X, ids in zip(COIN_SHORT_NAME, X_list, ids_list):
            # 存 X
            np.save(f"{INPUT_PATH}/X_input/price_classifier/{coin_short_name}/{coin_short_name}_{MODEL_NAME[1]}_X_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}{SUFFIX_CLASSIFIER_1}.npy", X)

            # 存 ids
            with open(f"{INPUT_PATH}/X_input/price_classifier/{coin_short_name}/{coin_short_name}_{MODEL_NAME[1]}_ids_classifier_2{SUFFIX_FILTERED}{SUFFIX_AUGUST}.pkl", "wb") as f:
                pickle.dump(ids, f)

        print(f"\n✅ 已成功儲存至 {INPUT_PATH}/X_input/price_classifier/{coin_short_name}\n")

if __name__ == "__main__":
    main()
