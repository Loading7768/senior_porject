#!/usr/bin/env python3
import os
import numpy as np
import joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# === 類別符號對應（用於報告） ===
LABEL_SYMBOLS = {
    0: "🔴 大跌",
    1: "🟠 小跌",
    2: "⚪ 持平",
    3: "🟡 小漲",
    4: "🟢 大漲"
}

# === 設定參數 ===
MODEL_NAME = "rf"              # 'rf', 'logreg', 'sgd' 可選
TRAIN_TYPE = "filtered"        # 'filtered' or 'non_filtered'
USE_CLASSIFIER_1 = False       # True = 有用 classifier_1 篩特徵；False = 沒用

# === 模型名稱轉換 ===
if MODEL_NAME == "logreg":
    TRAIN_NAME = "logistic_regression"
elif MODEL_NAME == "rf":
    TRAIN_NAME = "random_forest"
else:
    TRAIN_NAME = "SGD"

# === 標籤與路徑 TAG 組合 ===
train_type_tag = "" if TRAIN_TYPE == "filtered" else "_non_filtered"
classifier_tag = "" if USE_CLASSIFIER_1 else "_non_classifier_1"
DATA_PATH = f"../data/ml/dataset/final_input/price_classifier/{TRAIN_NAME}"

# === 自動產生檔案路徑 ===
X_train_path = f"{DATA_PATH}/{MODEL_NAME}_X_train_classifier_2{train_type_tag}{classifier_tag}.npy"
X_test_path  = f"{DATA_PATH}/{MODEL_NAME}_X_test_classifier_2{train_type_tag}{classifier_tag}.npy"
Y_train_path = f"{DATA_PATH}/{MODEL_NAME}_Y_train_classifier_2{train_type_tag}.npy"
Y_test_path  = f"{DATA_PATH}/{MODEL_NAME}_Y_test_classifier_2{train_type_tag}.npy"
IDS_train_path = f"{DATA_PATH}/{MODEL_NAME}_ids_train_classifier_2{train_type_tag}.pkl"
IDS_test_path  = f"{DATA_PATH}/{MODEL_NAME}_ids_test_classifier_2{train_type_tag}.pkl"

# === [1/6] 載入資料 ===
print("👉 [1/6] 載入資料中...")
X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
Y_train = np.load(Y_train_path)
Y_test = np.load(Y_test_path)
print(f"   ✔ X_train={X_train.shape}, X_test={X_test.shape}")
print(f"   ✔ Y_train={Y_train.shape}, Y_test={Y_test.shape}")


# === [3/6] 建立模型 ===
print("👉 [3/6] 建立 Random Forest 模型...")
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=40,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
print("   ✔ 模型建立完成")

# === [4/6] 開始訓練 ===
print("👉 [4/6] 開始訓練...")
model.fit(X_train, Y_train)
print("   ✔ 訓練完成")

# === [5/6] 預測與評估 ===
print("👉 [5/6] 預測與評估中...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(Y_train, y_train_pred)
test_acc = accuracy_score(Y_test, y_test_pred)

target_names = [LABEL_SYMBOLS[i] for i in range(5)]
report = []
report.append("=== 訓練報告 (Training Report) ===")
report.append(f"模型參數: n_estimators=400, max_depth=40, min_samples_leaf=5, class_weight=balanced")
report.append(f"訓練準確率 (Train) = {train_acc:.4f}")
report.append(f"測試準確率 (Test) = {test_acc:.4f}")
report.append("\n=== Train Classification Report ===")
report.append(classification_report(Y_train, y_train_pred, digits=3, target_names=target_names, labels=[0,1,2,3,4]))
report.append("\n=== Test Classification Report ===")
report.append(classification_report(Y_test, y_test_pred, digits=3, target_names=target_names, labels=[0,1,2,3,4]))

print("\n".join(report))

# === [6/6] 儲存模型與報告 ===
print("👉 [6/6] 儲存中...")
SAVE_DIR = f"../data/ml/models/classification/classifier2/{TRAIN_NAME}"
REPORT_DIR = f"../outputs/classification_report/{TRAIN_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

model_path = f"{SAVE_DIR}/{MODEL_NAME}_classifier2{train_type_tag}{classifier_tag}.joblib"
txt_path   = f"{REPORT_DIR}/{MODEL_NAME}_classifier2{train_type_tag}{classifier_tag}_results.txt"

joblib.dump(model, model_path)
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"   ✔ 模型已儲存：{model_path}")
print(f"   ✔ 報告已輸出：{txt_path}")
