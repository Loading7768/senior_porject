from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, LearningCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import loguniform, uniform

import json
import numpy as np
import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
from config import COIN_SHORT_NAME

# ----------------------------------------paths----------------------------------------
FEATURE_VECTOR_PATH = '../data/keyword/machine_learning/feature_vector.npy'
FEATURE_NAME_PATH = '../data/keyword/machine_learning/feature_name.json'
PRICE_VECTOR_PATH = '../data/coin_price/price_diff.npy'
OUTPUT_PATH = '../data/ml/regression'
FIGURE_PATH = '../outputs/figures/ml/regression'
# ----------------------------------------paths----------------------------------------

def prepare_data():
    # load dataset
    with open(FEATURE_NAME_PATH, 'r', encoding='utf-8') as file:
        feature_names = json.load(file)
    X = np.load(FEATURE_VECTOR_PATH)
    Y = np.load(PRICE_VECTOR_PATH)

    # split data into training/testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # standardize features (SVR 對 scale 很敏感)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test, feature_names

def tune_hyperparam(X_train, Y_train):
    model = SVR()

    # SVR 超參數範圍
    param_dist = {
        'C': loguniform(1e-5, 1e5),         # 正則化參數
        'epsilon': uniform(0.00001, 1.0),      # epsilon-insensitive tube
        'kernel': ['linear', 'rbf', 'poly'],# kernel type
        'gamma': ['scale', 'auto']          # 只對 rbf/poly 有效
    }

    search = RandomizedSearchCV(
        model, param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42
    )
    search.fit(X_train, Y_train)

    print(f'Best Hyperparameters: {search.best_params_}')
    print(f'Best CV MSE: {-search.best_score_}')

    return search.best_estimator_

def train_and_evaluate(best_model, X_train, X_test, Y_train, Y_test):
    best_model.fit(X_train, Y_train)

    Y_pred_train = best_model.predict(X_train)
    Y_pred_test = best_model.predict(X_test)

    train_mse = mean_squared_error(Y_train, Y_pred_train)
    test_mse = mean_squared_error(Y_test, Y_pred_test)
    test_r2 = r2_score(Y_test, Y_pred_test)

    print(f'\nTrain MSE: {train_mse}')
    print(f'Test MSE: {test_mse}')
    print(f'Test R^2: {test_r2}')

def validation(best_model, X_train, X_test, Y_train, Y_test):
    display = LearningCurveDisplay.from_estimator(
        best_model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    display.plot()
    os.makedirs(FIGURE_PATH, exist_ok=True)
    plt.savefig(f'{FIGURE_PATH}/svr_learning_curve.png')
    plt.show()

def main():
    print('Loading data...')
    X_train, X_test, Y_train, Y_test, feature_names = prepare_data()
    print('\nTunning hyperparameters...')
    best_model = tune_hyperparam(X_train, Y_train)
    print('\nTraining model...')
    train_and_evaluate(best_model, X_train, X_test, Y_train, Y_test)
    print('\nPlotting learning curve...')
    validation(best_model, X_train, X_test, Y_train, Y_test)

if __name__ == '__main__':
    main()