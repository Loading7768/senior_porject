import numpy as np
import pickle
import pandas as pd

'''可修改參數'''
COIN_SHORT_NAME = ["DOGE", "PEPE", "TRUMP"]

MODEL_NAME = "logreg"

INPUT_PATH = "../data/ml/dataset"

INPUT_FIRST_CLASSIFIER_PATH = "../data/ml/classification/logistic_regression"

OUTPUT_PATH = "../data/ml/dataset"

MERGE_CLASSIFIER_1_RESULT = True
'''可修改參數'''

def merge():
    X = []
    Y = []
    all_coin_dates = set()  # 用集合自動去重
    ids_all_coin = []

    # 將不同幣種的 X, Y 分別讀取進來
    for coin_short_name in COIN_SHORT_NAME:
        print(f"\n🚩 正在處理 {coin_short_name} ...")

        # --- 讀取 X ---
        X_diff_past = np.load(f"{INPUT_PATH}/coin_price/{coin_short_name}_price_diff_past5days.npy")  # 讀取 前面幾天 的 價差、價錢變化率
        X_XGBoost = np.load(f"{INPUT_PATH}/coin_price/{coin_short_name}_XGBoost_features.npy")  # 讀取 XBGoost 所使用的 features
        X_first_classifier = np.load(f"{INPUT_FIRST_CLASSIFIER_PATH}/{coin_short_name}_{MODEL_NAME}_classifier_1_result.npy")  # 讀取 第一個分類器 預測的結果
        
        # --- 讀取 X 的日期參考資料 ---
        XGBoost_dates = np.loadtxt(f"{INPUT_PATH}/coin_price/{coin_short_name}_XGBoost_dates.txt", dtype=str)  # 讀取 XBGoost 所使用的 dates
        with open(f"{INPUT_PATH}/keyword/{coin_short_name}_ids.pkl", "rb") as f:   # 讀取一開始訓練用的 ids
            ids = pickle.load(f)
        
        all_coin_dates.update([(c, d) for (c, d, _) in ids])  # 只取 (coin, date) 加入集合

        # 先把 all_coin_dates 只保留當前幣種的日期
        current_coin_dates = set([d for (c, d) in all_coin_dates if c == coin_short_name])

        # print("X_XGBoost.shape", X_XGBoost.shape)
        # print("XGBoost_dates[-10:]", XGBoost_dates[-10:])
        # print("current_coin_dates：", sorted(current_coin_dates))
        # print("len(current_coin_dates)：", len(current_coin_dates))

        # 建立 mask，只保留在 all_coin_dates 裡的日期
        mask = [d in current_coin_dates for d in XGBoost_dates]

        # 用 mask 過濾 X_XGBoost 與 XGBoost_dates
        X_XGBoost = X_XGBoost[mask]
        XGBoost_dates = XGBoost_dates[mask]  # 為了看 X_XGBoost 有沒有刪正確

        current_coin_ids = set([(c, d) for (c, d) in all_coin_dates if c == coin_short_name])
        # ids_all_coin += sorted(current_coin_ids)
        print(f"去掉重複日期後 {coin_short_name} 的 (coin, date) 數量: {len(current_coin_ids)}\n")
        print(f"{coin_short_name} 的 XGBoost 相關特徵日期的後 10 天：(用來檢查 X_XGBoost 有沒有取正確 DOGE、TRUMP前面會少 13 天)\n{XGBoost_dates[-10:]}\n")

        # print("X_diff_past.shape:", X_diff_past.shape)
        # print("X_XGBoost.shape:", X_XGBoost.shape)
        # print("X_first_classifier.shape:", X_first_classifier.shape)

        # --- 讀取 Y ---
        Y_single_coin = np.load(f"{INPUT_PATH}/coin_price/{coin_short_name}_price_diff_original.npy")  # 讀取 明天 的價錢變化率 (price_diff_rate_tomorrow)
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
        X_diff_past = X_diff_past[-min_len:]
        X_XGBoost = X_XGBoost[-min_len:]
        X_first_classifier = X_first_classifier[-min_len:]
        Y_single_coin = Y_single_coin[-min_len:]
        ids_all_coin += (sorted(current_coin_ids)[-min_len:])

        print(f"目前 ids_all_coin (要輸出的 ids) 內容：(應該三個幣種都要長一樣)\n{ids_all_coin[:10]}\n")
        print(f"ids_all_coin (要輸出的 ids) 的長度：{len(ids_all_coin)}\n")

        # --- 合併特徵 ---
        if MERGE_CLASSIFIER_1_RESULT:
            X_single_coin = np.hstack([X_diff_past, X_XGBoost, X_first_classifier.reshape(-1, 1)])
        else:
            X_single_coin = np.hstack([X_diff_past, X_XGBoost])
        
    
        # --- 存進總集合 ---
        X.append(X_single_coin)
        Y.append(Y_single_coin)

    # --- 把三個幣種合併成一個大陣列 ---
    X = np.vstack(X)
    Y = np.concatenate(Y)

    print("\n✅ 已經完成合併\n")

    return X, Y, ids_all_coin

def export_to_csv(X, Y, ids, output_path=f"{MODEL_NAME}_merged_dataset.csv"):
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
    X, Y, ids = merge()

    print("len(ids) =", len(ids))
    print("X.shape =", X.shape)
    print("Y.shape =", Y.shape)

    # 用法
    export_to_csv(X, Y, ids, f"{OUTPUT_PATH}/{MODEL_NAME}_merged_dataset.csv")

    print("🚩 打亂前：")
    print("\nX 預覽：\n", X[:10])
    print("\nY 預覽：\n", Y[:10])
    print("\nids 預覽：\n", ids[:10])

    # --- 打亂 X, Y, ids ---
    rng = np.random.default_rng(42)  # 可自訂種子
    indices = np.arange(Y.shape[0])
    rng.shuffle(indices)

    # print("Before shuffle:", Y[0], ids[0])
    # print("Index mapping:", indices[:10])  # 看前10筆打亂順序
    # print("After shuffle:", Y[indices[0]], ids[indices[0]])

    # k = 619  # 舊的 index
    # print("Original:", Y[k], ids[k])
    # print("Shuffled:", Y[indices.tolist().index(k)], ids[indices.tolist().index(k)])

    X = X[indices]
    Y = Y[indices]
    ids = np.array(ids)[indices]

    print("\n🚩 打亂後：")
    print("\nX 預覽：\n", X[:10])
    print("\nY 預覽：\n", Y[:10])
    print("\nids 預覽：\n", ids[:10])

    # 儲存
    np.save(f"{OUTPUT_PATH}/{MODEL_NAME}_X_classifier_2.npy", X)
    np.save(f"{OUTPUT_PATH}/{MODEL_NAME}_Y_classifier_2.npy", Y)
    with open(f"{OUTPUT_PATH}/{MODEL_NAME}_ids_classifier_2.pkl", 'wb') as file:
        pickle.dump(ids, file)  # 這裡只會存 ('coin', 'date') 且每個日期只有一筆

    print(f"\n✅ 已成功儲存至 {OUTPUT_PATH}\n")

if __name__ == "__main__":
    main()
