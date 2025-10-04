import json
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path
import os

from sklearn.model_selection import train_test_split



'''可修改參數'''
COIN_SHORT_NAME = ['DOGE', 'PEPE', 'TRUMP']

COIN_DELETE_DATE = [13, 0, 12]  # 每個幣種要刪除的天數

MODEL_SHORT_NAME = "logreg"  # "logreg" "rf" "sgd"

MODEL_PATH_NAME = "logistic_regression"  # "logistic_regression" "random_forest" "SGD"

IS_FILTERED = True  # 看是否有分 normal 與 bot
'''可修改參數'''

SUFFIX_FILTERED = "" if IS_FILTERED else "_non_filtered"
LATEX_SUFFIX_FILTERED = "_filtered" if IS_FILTERED else "_non_filtered"

save_json_path = f"../outputs/classification_report/{MODEL_PATH_NAME}"
os.makedirs(save_json_path, exist_ok=True)




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



def Latex(report_train, report_test):
    classes = ['大跌', '小跌', '持平', '小漲', '大漲']

    latex_str = r"""\begin{table}[H]
\centering
{\fontsize{12.5}{16}\selectfont
\begin{tabular}{c|ccc|ccc}
& \multicolumn{3}{c|}{Train set} & \multicolumn{3}{c}{Test set} \\
\hline
Class & Precision & Recall & F1-score & Precision & Recall & F1-score \\
\hline
"""

    for cls in classes:
        train_prec = report_train[cls]['precision']
        train_rec = report_train[cls]['recall']
        train_f1 = report_train[cls]['f1-score']
        
        test_prec = report_test[cls]['precision']
        test_rec = report_test[cls]['recall']
        test_f1 = report_test[cls]['f1-score']
        
        latex_str += f"{cls} & {train_prec:.2f} & {train_rec:.2f} & {train_f1:.2f} & {test_prec:.2f} & {test_rec:.2f} & {test_f1:.2f} \\\\\n"

    # 加上 Macro avg
    train_macro = report_train['macro avg']
    test_macro = report_test['macro avg']
    latex_str += r"\hline" + "\n"
    latex_str += f"Macro avg & {train_macro['precision']:.2f} & {train_macro['recall']:.2f} & {train_macro['f1-score']:.2f} & {test_macro['precision']:.2f} & {test_macro['recall']:.2f} & {test_macro['f1-score']:.2f} \\\\\n"

    latex_str += r"""\end{tabular}
}
\caption{"""
    latex_str += MODEL_SHORT_NAME.capitalize()
    
    latex_str += r""" 以日期為單位的訓練與測試準確度}
\label{tab:classifier_1_report_"""
    latex_str += f"{MODEL_SHORT_NAME}{LATEX_SUFFIX_FILTERED}"
    latex_str += r"""}
\end{table}"""

    # 印出 LaTeX
    print(latex_str)

    # 可選：存成檔案
    with open(f"{save_json_path}/{MODEL_SHORT_NAME}_report{SUFFIX_FILTERED}.tex", "w", encoding="utf-8") as f:
        f.write(latex_str)




y_true_final, y_pred_final = [], []
for csn, delete in zip(COIN_SHORT_NAME, COIN_DELETE_DATE):
    print(f"\n目前正在執行 {csn} ...\n")
    Y_TRUE_PATH = Path(f'../data/ml/dataset/coin_price/{csn}_price_diff_original{SUFFIX_FILTERED}.npy')
    Y_PRED_PATH = Path(f'../data/ml/classification/{MODEL_PATH_NAME}/{csn}_{MODEL_SHORT_NAME}_classifier_1_result{SUFFIX_FILTERED}.npy')

    y_true = categorize_array_multi(np.load(Y_TRUE_PATH)).tolist()
    y_pred = np.load(Y_PRED_PATH).tolist()

    print("🚩 刪除資料前")
    print("len(y_true):", len(y_true))
    print("len(y_pred):", len(y_pred))
    print("y_true[:10]:", y_true[:30])
    print("y_pred[:10]:", y_pred[:30])

    y_true = y_true[delete:]
    y_pred = y_pred[delete:]
    y_true_final += y_true
    y_pred_final += y_pred

    print("🚩 刪除資料後")
    print("len(y_true):", len(y_true))
    print("len(y_pred):", len(y_pred))
    print("y_true[:10]:", y_true[:30])
    print("y_pred[:10]:", y_pred[:30])


# --- 打亂 X, Y, ids ---
rng = np.random.default_rng(42)  # 可自訂種子
indices = np.arange(len(y_pred_final))
rng.shuffle(indices)


y_true_final = [y_true_final[i] for i in indices]
y_pred_final = [y_pred_final[i] for i in indices]


print("🚩 打亂後")
print("len(y_true_final):", len(y_true_final))
print("len(y_pred_final):", len(y_pred_final))
print("y_true_final[:10]:", y_true_final[:30])
print("y_pred_final[:10]:", y_pred_final[:30])


y_true_train, y_true_test, y_pred_train, y_pred_test = train_test_split(
    y_true_final, y_pred_final, test_size=0.2, random_state=42, shuffle=True
)


print(classification_report(
    y_true_train, y_pred_train,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲']
))

print(classification_report(
    y_true_test, y_pred_test,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲']
))


# --- 生成分類報告 dict ---
report_train = classification_report(
    y_true_train, y_pred_train,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲'],
    output_dict=True   # <-- 這裡把報告轉成 dict
)

report_test = classification_report(
    y_true_test, y_pred_test,
    digits=3,
    target_names=['大跌', '小跌', '持平', '小漲', '大漲'],
    output_dict=True
)

Latex(report_train, report_test)


# --- 存成 JSON ---
save_json_path = f"../outputs/classification_report/{MODEL_PATH_NAME}"
os.makedirs(save_json_path, exist_ok=True)

with open(f"{save_json_path}/{MODEL_SHORT_NAME}_train_report{SUFFIX_FILTERED}.json", "w", encoding="utf-8") as f:
    json.dump(report_train, f, ensure_ascii=False, indent=4)

with open(f"{save_json_path}/{MODEL_SHORT_NAME}_test_report{SUFFIX_FILTERED}.json", "w", encoding="utf-8") as f:
    json.dump(report_test, f, ensure_ascii=False, indent=4)

print("\n✅ JSON 檔案已存好")