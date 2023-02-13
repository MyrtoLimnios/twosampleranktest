# Implementation of RankBost
# Uses classifiers for scikit-learn: AdaBoostClassifier
# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

import numpy as np

from sklearn.ensemble import AdaBoostClassifier

class RankB(AdaBoostRegressor):
    def __init__(self):
        super(RankB, self).__init__()

    def fit(self, X, y):
        X_trans, y_trans = pairwise(X, y)
        fit_ = super(RankB, self).fit(X_trans, y_trans)
        return fit_


    def predict(self, X):
        pred = super(RankB, self).predict(X)
        return pred

    def predict_proba(self, X):
        pred_proba = super(RankB, self).predict_proba(X)
        return pred_proba

    def score(self, X, y):
        X_trans, y_trans = pairwise(X, y)
        return np.mean(super(RankB, self).predict(X_trans) == y_trans)


def pairwise(X, y):
    X_new = []
    y_new = []
    y = np.asarray(y)
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()
