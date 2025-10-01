from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, LearningCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import loguniform, uniform
from scipy.sparse import csr_matrix

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
N_SAMPLES = 1000000
N_RUNS = 1
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
    X_TRAIN_PATH = Path(f'../data/ml/dataset/X_train{FILTERED_SUFFIX}.npz')
    X_TEST_PATH = Path(f'../data/ml/dataset/X_test{FILTERED_SUFFIX}.npz')
    Y_TRAIN_PATH = Path(f'../data/ml/dataset/Y_train{FILTERED_SUFFIX}.npz')
    Y_TEST_PATH = Path(f'../data/ml/dataset/Y_test{FILTERED_SUFFIX}.npz')

    # load dataset
    X_train = sparse.load_npz(X_TRAIN_PATH)
    X_test = sparse.load_npz(X_TEST_PATH)
    Y_train = categorize_array_multi(np.load(Y_TRAIN_PATH)['Y'])
    Y_test = categorize_array_multi(np.load(Y_TEST_PATH)['Y'])
    
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test

def get_random_samples_sparse_stratified(X: csr_matrix, y: np.ndarray, seed: int = 42):
    """
    X: csr_matrix
    y: np.ndarray, shape=(N,)  多類別標籤
    """
    global N_SAMPLES, ENABLE_SAMPLING
    n_total = X.shape[0]

    if N_SAMPLES == 0:
        print(f"[INFO] 不做 random sampling，使用所有樣本數: {n_total} 筆")
        ENABLE_SAMPLING = False
        return [(X, y)] * N_RUNS

    classes = np.unique(y)
    n_classes = len(classes)
    if N_SAMPLES < n_classes:
        raise ValueError(f"樣本數 {N_SAMPLES} 太少，無法平均分配到每個類別 ({n_classes})")
    
    samples_per_class = N_SAMPLES // n_classes

    # 建立索引字典
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    samples = []
    for run in range(N_RUNS):
        np.random.seed(seed + run)
        selected_indices = []

        for c in classes:
            idx_list = class_indices[c]
            if len(idx_list) <= samples_per_class:
                # 如果該類別數量不夠，就全部拿
                selected_indices.extend(idx_list)
            else:
                selected_indices.extend(np.random.choice(idx_list, samples_per_class, replace=False))

        # 如果總數少於 N_SAMPLES，從剩餘樣本補足
        if len(selected_indices) < N_SAMPLES:
            remaining_idx = list(set(range(n_total)) - set(selected_indices))
            remaining_needed = N_SAMPLES - len(selected_indices)
            selected_indices.extend(np.random.choice(remaining_idx, remaining_needed, replace=False))

        np.random.shuffle(selected_indices)  # 打亂順序
        X_sample = X[selected_indices]
        y_sample = y[selected_indices]
        samples.append((X_sample, y_sample))

        # === 新增：統計類別數量與比例 ===
        unique, counts = np.unique(y_sample, return_counts=True)
        total = len(y_sample)
        print(f"\n[INFO] Run {run}: Stratified sample X_train={X_sample.shape}, y_train={y_sample.shape}")
        for cls, cnt in zip(unique, counts):
            pct = cnt / total * 100
            print(f"   Class {cls}: {cnt} samples ({pct:.2f}%)")

    return samples

def train_and_evaluate(X_train_sample, Y_train_sample, X_train, X_test, Y_train, Y_test):
    '''
    Train model, generate prediction for the whole dataset and data from August.
    '''
    if USE_NON_FILTERED:
        knn = KNeighborsClassifier(n_jobs=-1, n_neighbors=65, weights='distance', metric='euclidean')
    else: 
        knn = KNeighborsClassifier(n_jobs=-1)

    # param_grid = {
    #     'n_neighbors': [65, 66, 67, 68],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan'],
    # }
    # grid_search = GridSearchCV(knn, param_grid, cv=2, scoring='accuracy', n_jobs=-1, verbose=True)
    # grid_search.fit(X_train_sample, Y_train_sample)

    # print('Best params:', grid_search.best_params_)
    # print('Best CV score:', grid_search.best_score_)

    knn.fit(X_train_sample, Y_train_sample)

    MODEL_PATH = Path(f'../data/ml/models/classification/KNN/knn{FILTERED_SUFFIX}.pkl')
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    joblib.dump(knn, MODEL_PATH)

    train_score = knn.score(X_train_sample, Y_train_sample)
    print('Train accuracy:', train_score)
    test_score = knn.score(X_test, Y_test)
    print('Test accuracy:', test_score)
    print("\nClassification Report (Test set):")
    print(classification_report(Y_test, knn.predict(X_test)))

    # generate prediction for the whole dataset
    pre_trained = []
    pre_august = []
    pre_trained += knn.predict(X_train).tolist()
    pre_trained += knn.predict(X_test).tolist()

    # generate prediction for data from August
    for csn in ['DOGE', 'PEPE', 'TRUMP']:
        X_AUGUST_PATH = f'../data/ml/dataset/keyword/{csn}_X_sparse{FILTERED_SUFFIX}_202508.npz'
        Y_AUGUST_PATH = f'../data/ml/dataset/coin_price/{csn}_price_diff{FILTERED_SUFFIX}_202508.npy'

        X_august = sparse.load_npz(X_AUGUST_PATH)
        Y_august = categorize_array_multi(np.load(Y_AUGUST_PATH))

        august_score = knn.score(X_august, Y_august)
        print(f'{csn} August accuracy: {august_score}')

        pre_august = knn.predict(X_august)

    return pre_trained, pre_august

def counting_votes(data):
    count = defaultdict(lambda: defaultdict(int))
    for d in data:
        count[d[1]][d[3]] += 1
    count = sorted(count.items(), key= lambda x:x[0])

    output = []
    for _, cat_counter in count:
        pre_cat, vote = '0', 0
        for cat, counter in cat_counter.items():
            if counter > vote:
                pre_cat, vote = cat, counter

        output.append(pre_cat)

    return output

def sort_and_save(pre_trained, pre_august):
    IDS_TRAIN_PATH = f'../data/ml/dataset/ids_train{FILTERED_SUFFIX}.pkl'
    IDS_TEST_PATH = f'../data/ml/dataset/ids_test{FILTERED_SUFFIX}.pkl'
    OUTPUT_PATH = '../data/ml/classification/KNN'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # load ids and append prediction
    ids = []
    for path in [IDS_TRAIN_PATH, IDS_TEST_PATH]:
        with open(path, 'rb') as file:
            ids += pickle.load(file)
    for i, pt in zip(ids, pre_trained):
        i.append(pt)

    for csn in ['DOGE', 'PEPE', 'TRUMP']:
        data = [i for i in ids if i[0] == csn]
        output = counting_votes(data)
        np.save(f'{OUTPUT_PATH}/{csn}_knn_classifier_1_result{FILTERED_SUFFIX}.npy', output)

        IDS_AUGUST_PATH = f'../data/ml/dataset/{csn}_ids{FILTERED_SUFFIX}_202508.pkl'

        with open(IDS_AUGUST_PATH, 'rb') as file:
            ids_august = pickle.load(file)
        for i, pa in zip(ids_august, pre_august):
                i.append(pa)

        data = [i for i in ids if i[0] == csn]
        output = counting_votes(data)
        np.save(f'{OUTPUT_PATH}/{csn}_knn_classifier_1_result{FILTERED_SUFFIX}_202508.npy', output)

def main():
    print(f'Using data from {'all' if USE_NON_FILTERED else 'normal'} tweets.')

    print(f'Loading data of {N_SAMPLES} samples...')
    X_train, X_test, Y_train, Y_test = prepare_data()

    samples = get_random_samples_sparse_stratified(X_train, Y_train) 
    for s in samples:
        X_train_sample, Y_train_sample = s[0], s[1]
        pre_trained, pre_august = train_and_evaluate(X_train_sample, Y_train_sample, X_train, X_test, Y_train, Y_test)
        sort_and_save(pre_trained, pre_august)

if __name__ == '__main__':
    main()          
