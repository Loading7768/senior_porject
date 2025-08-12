from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import loguniform

import json
import numpy as np
import os

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
# ----------------------------------------paths----------------------------------------

def prepare_data():
    # load dataset
    feature_names = []
    with open(FEATURE_NAME_PATH, 'r', encoding='utf-8') as file:
        feature_names = json.load(file)
    feature_vector = np.load(FEATURE_VECTOR_PATH)
    price_vector = np.load(PRICE_VECTOR_PATH)

    '''
    split data into:
    - 60% training
    - 20% validation
    - 20% testing
    '''
    X_temp, X_test, Y_temp, Y_test = train_test_split(feature_vector, price_vector, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=42)

    # standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, feature_names

def tune_hyperparam(X_train, Y_train):
    model = BayesianRidge()

    # parameter distribution
    param_dist = {
        'alpha_1': loguniform(1e-7, 1e-3),
        'alpha_2' : loguniform(1e-7, 1e-3),
        'lambda_1' : loguniform(1e-7, 1e-3),
        'lambda_2' : loguniform(1e-7, 1e-3),
    }

    search = RandomizedSearchCV(
        model, param_dist, n_iter=50, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42,
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

def validation():
    pass

def main():
    X_train, X_val, X_test, Y_train, Y_val, Y_test, feature_names = prepare_data()
    best_model = tune_hyperparam(X_train,Y_train)
    train_and_evaluate(best_model, X_train, X_test, Y_train, Y_test)

    # coeficients = model_training(feature_vector, price_vector)
    
    # feature_names_importance = {name: coef for name, coef in zip(feature_names, coeficients)}
    # sorted_importance = sorted(feature_names_importance.items(), key=lambda x: x[1], reverse=True)
    # sorted_dict = {k: v for k, v in sorted_importance}
    
    # os.makedirs(OUTPUT_PATH, exist_ok=True)
    # with open(OUTPUT_PATH + '/bayesian_importance.json', 'w', encoding='utf-8') as file:
    #     json.dump(sorted_dict, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()