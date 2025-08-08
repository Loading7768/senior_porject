from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import json

# 取得 ML 的 feature_vector
features_path = "../data/keyword/machine_learning"
features_vector = np.load(f"{features_path}/feature_vector.npy")
with open(f"{features_path}/feature_name.json", "r", encoding="utf-8-sig") as jsonfile:
    features_name = json.load(jsonfile)

# 取得 price 的 csv 檔
price_path = "../data/coin_price"
df = pd.read_csv(f"{price_path}/current_tweet_price_output.csv")
df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")  # 把 date 欄位轉成日期格式

# 取得 price_diff 的 npy 檔
price_diff = np.load(f"{price_path}/price_diff.npy")  # 準備一個 Y 陣列

# 建立 target label：上漲為 1，否則為 0（二元分類）
labels = (price_diff > 0).astype(int)

# 把當天沒有抓到推文的日期存起來
unprocessed_dates = []
for i in range(len(df)):
    if df.loc[i, "has_tweet"] == False:
        unprocessed_dates.append(df.loc[i, "date"].strftime("%Y/%m/%d"))

# 最後一筆也無法計算（因為沒「明天」）
# unprocessed_dates.append(df.loc[len(df)-1, "date"].strftime("%Y/%m/%d"))

# 過濾 features_vector 和 labels
X = features_vector
Y = labels

# 使用 L1 正則化訓練 Logistic Regression
model = LogisticRegression(
    penalty='l1', 
    solver='saga',  # 'liblinear'  'saga'
    max_iter=5000,  # 迭代次數
    verbose=1
)
model.fit(X, Y)

# 顯示每個關鍵字的係數（由大到小）
coefficients = pd.Series(model.coef_[0], index=features_name)
print(coefficients.sort_values(ascending=False))


# 額外印出哪些日期是被排除的
print("\n被排除的日期（沒有推文或無法計算價格變化）:")
print(unprocessed_dates)

# 將係數轉成 dict
coeff_dict = coefficients.to_dict()

# 存成 JSON 檔
output_path = "../data/ml/classification/logistic_regression_keyword_coefficients.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(coeff_dict, f, ensure_ascii=False, indent=4)

print(f"已存成 JSON：{output_path}")
