from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, LearningCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from scipy.stats import loguniform

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
# ----------------------------------------paths----------------------------------------

def prepare_data():
    # load dataset
    feature_names = []
    with open(FEATURE_NAME_PATH, 'r', encoding='utf-8') as file:
        feature_names = json.load(file)
    X = np.load(FEATURE_VECTOR_PATH)
    Y = np.load(PRICE_VECTOR_PATH)

    '''
    split data into:
    - 80% training
    - 20% testing
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # standardize features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    return X_train, X_test, Y_train, Y_test, feature_names

def tune_hyperparam(X_train, Y_train):
    model = BayesianRidge()

    # parameter distribution
    param_dist = {
        'alpha_1': loguniform(1e-8, 1e-1),
        'alpha_2' : loguniform(1e-8, 1e-1),
        'lambda_1' : loguniform(1e-8, 1e-1),
        'lambda_2' : loguniform(1e-8, 1e-1),
    }

    search = RandomizedSearchCV(
        model, param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error',
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

def validation(best_model, X_train, X_test, Y_train, Y_test):
    # train_sizes, train_scores, val_scores = learning_curve(
    #     best_model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error',
    #     train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    # )
    display = LearningCurveDisplay.from_estimator(
        best_model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    display.plot()
    plt.show()

def main():
    print('Loading data...')
    X_train, X_test, Y_train, Y_test, feature_names = prepare_data()
    print('\nTunning hyperparameters...')
    best_model = tune_hyperparam(X_train,Y_train)
    print('\nTraining model...')
    train_and_evaluate(best_model, X_train, X_test, Y_train, Y_test)
    print('\nPlotting learning curve...')
    validation(best_model, X_train, X_test, Y_train, Y_test)

    # coeficients = model_training(feature_vector, price_vector)
    
    # feature_names_importance = {name: coef for name, coef in zip(feature_names, coeficients)}
    # sorted_importance = sorted(feature_names_importance.items(), key=lambda x: x[1], reverse=True)
    # sorted_dict = {k: v for k, v in sorted_importance}
    
    # os.makedirs(OUTPUT_PATH, exist_ok=True)
    # with open(OUTPUT_PATH + '/bayesian_importance.json', 'w', encoding='utf-8') as file:
    #     json.dump(sorted_dict, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()