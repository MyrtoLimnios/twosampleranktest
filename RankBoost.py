# Implementation of RankBoth with modified pairwise transformation
# Uses classifiers for scikit-learn: AdaBoostClassifier
# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr

import numpy as np

from sklearn.ensemble import AdaBoostClassifier

class RankB(AdaBoostClassifier):
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
    k = 0
    for pos_idx in np.where(y == 1)[0]:
        for all_idx in range(len(y)):
            X_new.append(X[pos_idx] - X[all_idx])
            y_new.append(y[all_idx])
        if y_new[-1] != (-1) ** k:
            y_new[-1] *= -1
            X_new[-1] *= -1
        k += 1
    return np.asarray(X_new), np.asarray(y_new)

