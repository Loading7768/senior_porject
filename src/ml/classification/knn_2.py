from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform, uniform
from scipy.sparse import csr_matrix

import json
import numpy as np
from scipy import sparse
import os
import matplotlib.pyplot as plt
import argparse
import joblib
from tqdm import tqdm
import math
from collections import defaultdict
import pickle

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME

# ----------------------------------------parameters----------------------------------------
FIRST_MODEL_NAME = 'logreg'
USE_NON_FILTERED = False
# ----------------------------------------parameters----------------------------------------
FILTERED_SUFFIX = '_non_filtered' if USE_NON_FILTERED else ''

def categorize_array_multi(Y, t1 = -0.0590, t2 = -0.0102, t3 = 0.0060, t4 = 0.0657):
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

    if np.any(Y == 0):  # 檢查是否有任何元素等於 0
        count = np.sum(Y == 0)
        print(f"共有 {count} 個 Y == 0")
        labels[Y == 0] = 4  # 為了校正TRUMP前兩天的價格相同 第一天設為大漲

    return labels

def prepare_data():
    '''
    args: feature selection arguments
    '''
    X_PATH = Path(f'../data/ml/dataset/{FIRST_MODEL_NAME}_X_classifier_2{FILTERED_SUFFIX}.npy')
    Y_PATH = Path(f'../data/ml/dataset/{FIRST_MODEL_NAME}_Y_classifier_2{FILTERED_SUFFIX}.npy')

    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    Y_train = categorize_array_multi(Y_train)
    Y_test  = categorize_array_multi(Y_test)

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test


def train_and_evaluate(X_train, X_test, Y_train, Y_test):
    if USE_NON_FILTERED:
        knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=19, weights='distance', metric='manhattan')
    else:
        knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=42, weights='uniform', metric='manhattan')
    # param_grid = {
    #     'n_neighbors': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan', 'minkowski']
    # }
    # grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=True)
    # grid_search.fit(X_train, Y_train)

    # print('Best params:', grid_search.best_params_)
    # print('Best CV score:', grid_search.best_score_)

    # best_model = grid_search.best_estimator_
    # test_score = best_model.score(X_test, Y_test)
    # print('Test accuracy:', test_score)

    knn.fit(X_train, Y_train)

    MODEL_PATH = Path(f'../data/ml/models/classification/KNN2/{FIRST_MODEL_NAME}_KNN_2{FILTERED_SUFFIX}.pkl')
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(knn, MODEL_PATH)

    train_score = knn.score(X_train, Y_train)
    test_score = knn.score(X_test, Y_test)
    clf_rpt = classification_report(Y_test, knn.predict(X_test))
    model_report = f'Train accuracy: {train_score}\nTest accuracy: {test_score}\n\nClassification Report (Test set):\n{clf_rpt}\n'
    print(model_report)

    pre_trained = []
    pre_trained += knn.predict(X_train).tolist()
    pre_trained += knn.predict(X_test).tolist()
    Y = np.concatenate((Y_train, Y_test), axis=0)
    pre_trained = [[pt, y] for pt, y in zip(pre_trained, Y)]

    august_accuracy = []
    pre_august_all = []
    for csn in ['DOGE', 'PEPE', 'TRUMP']:
        X_AUGUST_PATH = f'../data/ml/dataset/keyword/{csn}_{FIRST_MODEL_NAME}_X_classifier_2{FILTERED_SUFFIX}_202508.npy'
        Y_AUGUST_PATH = f'../data/ml/dataset/coin_price/{csn}_price_diff_original{FILTERED_SUFFIX}_202508.npy'

        X_august = np.load(X_AUGUST_PATH)
        Y_august = categorize_array_multi(np.load(Y_AUGUST_PATH))

        pre_august = knn.predict(X_august).tolist()

        august_score = knn.score(X_august, Y_august)
        august_accuracy.append(f'Accuracy: {august_score}')
        
        pre_august = [[pt, y] for pt, y in zip(pre_august, Y_august)]

        pre_august_all.append(pre_august)

    return pre_trained, model_report, pre_august_all, august_accuracy

def sort_and_save(pre_trained, model_report, pre_august_all, august_accuracy):
    cat = ["🔴","🟠","⚪","🟡","🟢"]
    OUTPUT_PATH = f'../data/ml/classification/KNN2'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    IDS_PATH = f'../data/ml/dataset/{FIRST_MODEL_NAME}_ids_classifier_2{FILTERED_SUFFIX}.pkl'

    with open(IDS_PATH, 'rb') as file:
        ids = pickle.load(file).tolist()

    ids = [i + pt for i, pt in zip(ids, pre_trained)]
    ids = sorted(ids, key=lambda x: (x[0], x[1]))

    csn = ['DOGE', 'PEPE', 'TRUMP']
    model_report += f'=={csn.pop(0)}==\n'
    for i in ids:
        model_report += f'{i[1]} → 預測: {cat[i[2]]}  真實: {cat[i[3]]}  結果: {'✅' if i[3] == i[2] else '❌'}\n'
        if i[1] == '2025-07-31' and csn:
            model_report += f'\n=={csn.pop(0)}==\n'

    with open(f'{OUTPUT_PATH}/{FIRST_MODEL_NAME}_knn_combined_classifier_2_results{FILTERED_SUFFIX}.txt', 'w', encoding='utf-8-sig') as file:
        file.write(model_report)

    august_report = ''
    for csn, pre_august in zip(['DOGE', 'PEPE', 'TRUMP'], pre_august_all):
        IDS_AUGUST_PATH = f'../data/ml/dataset/keyword/{csn}_{FIRST_MODEL_NAME}_ids_classifier_2{FILTERED_SUFFIX}_202508.pkl'

        with open(IDS_AUGUST_PATH, 'rb') as file:
            august_ids = pickle.load(file)
        august_ids = [list(i) for i in august_ids]
        august_ids = [i + pa for i, pa in zip(august_ids, pre_august)]

        august_report += f'=={csn}==\n'
        for i in august_ids:
            august_report += f'{i[1]} → 預測: {cat[i[2]]}  真實: {cat[i[3]]}  結果: {'✅' if i[3] == i[2] else '❌'}\n'
                
        august_report += f'\n{august_accuracy.pop(0)}\n\n'

    with open(f'{OUTPUT_PATH}/{FIRST_MODEL_NAME}_knn_combined_classifier_2_results{FILTERED_SUFFIX}_202508.txt', 'w', encoding='utf-8-sig') as file:
        file.write(august_report)

def main():
    print(f'Using results from {'all' if USE_NON_FILTERED else 'normal'} tweets.')

    print('Loading data...')
    X_train, X_test, Y_train, Y_test = prepare_data()
    pre_trained, model_report, pre_august_all, august_accuracy = train_and_evaluate(X_train, X_test, Y_train, Y_test)
    sort_and_save(pre_trained, model_report, pre_august_all, august_accuracy)

if __name__ == '__main__':
    main()          
