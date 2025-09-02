'''
'''
from scipy.sparse import save_npz, load_npz
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import pickle

# ----------------------------------------paths----------------------------------------
X_TRAIN_PATH = '../data/ml/dataset/X_train.npz'
Y_TRAIN_PATH = '../data/ml/dataset/Y_train.npy'
KEYWORD_PATH = '../data/ml/dataset/keyword/filtered_keywords.json'
TRAIN_ID_PATH = '../data/ml/dataset/keyword/ids_train.pkl'
# ----------------------------------------paths----------------------------------------

# # ----------parameters----------
# THRESHOLD = 1
# # ----------parameters----------

def load_data():
    X_train = load_npz(X_TRAIN_PATH)
    Y_train = np.load(Y_TRAIN_PATH)

    with open(KEYWORD_PATH, 'r', encoding='utf-8') as file:
        keywords = json.load(file)

    with open(TRAIN_ID_PATH, 'rb') as file:
        ids = pickle.load(file)

    return X_train, Y_train, keywords, ids

def radix_sort(X_train, Y_train, keywords):
    # keyword_tweet_frequency
    tf = X_train.sum(axis=0).A1
    keywords_tf = [(k, idx, tf) for k, idx, tf in zip(keywords, range(len(keywords)), tf)]
    keywords_tf = sorted(keywords_tf, key= lambda x: x[2])

    sorted_train = list(zip(X_train, Y_train))
    for _, idx, _ in tqdm(keywords_tf, desc="Radix sorting keywords"):
        has_keyword = []
        no_keyword = []
        for t in sorted_train:
            has_keyword.append(t) if t[0][0, idx] else no_keyword.append(t)

        sorted_train = has_keyword + no_keyword

    return sorted_train

def filter_data(sorted_train):
    # base case
    pivot = 0
    next = pivot
    scale = 1 if sorted_train[pivot][1] >= 0 else -1

    while pivot < len(sorted_train) - 1:
        next = next + 1
        print(next)
        if (sorted_train[next][0] != sorted_train[pivot][0]).nnz == 0:
            scale = scale + 1 if sorted_train[next][1] >= 0 else scale - 1
        else:
            if scale == 0:
                print(f'({pivot}, {next - 1})')

            pivot = next
            continue

def find_duplicate_rows(X_train, Y_train, ids):
    seen = defaultdict(list)

    for i in tqdm(range(X_train.shape[0]), desc='Finding the same rows'):
        row = X_train.getrow(i)
        # Represent row as tuple of (indices, values)
        key = (tuple(row.indices), tuple(row.data))
        seen[key].append(i)

    # Only keep groups with >1 element (duplicates)
    duplicates = {k: idxs for k, idxs in seen.items() if len(idxs) > 1}
    mask = [True] * X_train.shape[0]
    for _, idxs in duplicates.items():
        scales = [0, 0, 0, 0, 0]
        for i in idxs:
            for j in range(5):
                scales[j] = scales[j] + 1 if Y_train[i][j] >= 0 else scales[j] - 1

        nearly_eq = True
        for j in range(5):
            if abs(scales[j])/len(idxs) > 0.05:
                nearly_eq = False

        if nearly_eq:  
            for i in idxs:
                mask[i] = False

    ids = np.array(ids)
    
    print(f'{X_train[mask].shape[0]} ({X_train.shape[0] - X_train[mask].shape[0]})')
    return X_train[mask], Y_train[mask], ids[mask]
        
def save_result():
    pass

def main():
    print('Loading...')
    X_train, Y_train, keywords, ids = load_data()
    X_train_filtered, Y_train_filtered, ids_filtered= find_duplicate_rows(X_train, Y_train, ids)
    save_npz('../data/ml/dataset/X_train_filtered.npz', X_train_filtered)
    np.save('../data/ml/dataset/Y_train_filtered.npy', Y_train_filtered)
    ids_filtered = [tuple(i) for i in ids_filtered]
    with open('../data/ml/dataset/keyword/ids_train_filtered.pkl', 'wb') as file:
        pickle.dump(ids_filtered, file)

    # print('Sorting...')
    # sorted_train = radix_sort(X_train, Y_train, keywords)
    # print(len(sorted_train))
    # print('filtering...')
    # filter_data(sorted_train)

if __name__ == '__main__':
    main()