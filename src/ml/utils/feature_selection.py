"""
svr.py:
python -m ml.regression.svr --fs kbest-anova --k 200
python -m ml.regression.svr --fs kbest-mi --k 200
python -m ml.regression.svr --fs rfe-svr --n-features 100
python -m ml.regression.svr --fs sfm-tree
python -m ml.regression.svr --fs sfm-l1

random_forest.py:
python -m ml.classification.random_forest --fs kbest-chi2 --k 800
python -m ml.classification.random_forest --fs sfm-tree

bayesian.py:
python -m ml.regression.bayesian --fs sfm-bayes

logistic_regression.py:
python -m ml.classification.logistic_regression --fs kbest-chi2
python -m ml.classification.logistic_regression --fs kbest-anova
python -m ml.classification.logistic_regression --fs kbest-mi
python -m ml.classification.logistic_regression --fs sfm-l1
"""


from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVR

def make_selector(task: str, method: str, **kwargs):
    """
    task: 'clf' | 'reg'
    method (clf):
      - 'none' | 'kbest-chi2' | 'kbest-anova' | 'kbest-mi' | 'rfe-logreg' | 'sfm-tree' | 'sfm-l1'
    method (reg):
      - 'kbest-anova' | 'kbest-mi' | 'rfe-svr' | 'sfm-tree' | 'sfm-l1' | 'sfm-bayes'
    """

    if method == "none":
        return None

    # classification
    if task == "clf":
        if method == "kbest-chi2":   # 文本分類
            return SelectKBest(score_func=chi2, k=kwargs.get("k", 600))
        if method == "kbest-anova":  # 數值分類
            return SelectKBest(score_func=f_classif, k=kwargs.get("k", 600))
        if method == "kbest-mi":     # 非線性分類
            return SelectKBest(score_func=mutual_info_classif, k=kwargs.get("k", 600))
        if method == "rfe-logreg":   # Logistic RFE
            base = LogisticRegression(penalty="l2", solver="liblinear", max_iter=5000)
            return RFE(base, n_features_to_select=kwargs.get("n_features", 300), step=0.1)
        if method == "sfm-tree":     # RandomForest (clf)
            est = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
            return SelectFromModel(est, threshold="median")
        if method == "sfm-l1":       # L1 Logistic
            est = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000)
            return SelectFromModel(est)

    # regression

    if task == "reg":
        if method == "kbest-anova":  # f_regression
            return SelectKBest(score_func=f_regression, k=kwargs.get("k", 300))
        if method == "kbest-mi":     # mutual info regression
            return SelectKBest(score_func=mutual_info_regression, k=kwargs.get("k", 300))
        if method == "rfe-svr":      # SVR RFE
            base = LinearSVR(max_iter=5000, random_state=0)
            return RFE(base, n_features_to_select=kwargs.get("n_features", 100), step=0.1)
        if method == "sfm-tree":     # RandomForest (reg)
            est = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
            return SelectFromModel(est, threshold="median")
        if method == "sfm-l1":       # Lasso
            est = Lasso(alpha=0.001, max_iter=5000, random_state=0)
            return SelectFromModel(est)
        if method == "sfm-bayes":    # BayesianRidge
            est = BayesianRidge()
            return SelectFromModel(est)

    raise ValueError("unknown method")