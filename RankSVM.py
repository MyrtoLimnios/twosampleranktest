# Implementation of RankSVM with modified pairwise transformation
# Uses classifiers for scikit-learn: LinearSVC, LinearSVR and LogisticRegression
# author: Myrto Limnios // mail: myrto.limnios@ens-paris-saclay.fr


import numpy as np

from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


class RankSVM(LinearSVC):
    def __init__(self, loss, penalty,  fit_intercept, dual, tol, C):
        super(RankSVM, self).__init__(loss=loss, penalty=penalty,
                                      fit_intercept=fit_intercept, dual=dual, tol=tol, C=C)

    def fit(self, X, y):
        X_trans, y_trans = pairwise(X, y)
        fit_ = super(RankSVM, self).fit(X_trans, y_trans)
        return fit_


    def predict(self, X):
        pred = super(RankSVM, self).predict(X)
        return pred


    def score(self, X, y):
        X_trans, y_trans = pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)



class RankSVMR(LinearSVR):
    def __init__(self, loss, tol, C):
        super(RankSVMR, self).__init__(loss=loss, tol=tol, C=C)

    def fit(self, X, y):
        X_trans, y_trans = pairwise(X, y)-
        fit_ = super(RankSVMR, self).fit(X_trans, y_trans)
        return fit_


    def predict(self, X):
        pred = super(RankSVMR, self).predict(X)
        return pred

    def score(self, X, y):
        X_trans, y_trans = pairwise(X, y)
        return np.mean(super(RankSVMR, self).predict(X_trans) == y_trans)



class RankSVML(LogisticRegression):
    def __init__(self, penalty, tol, C, solver):
        super(RankSVML, self).__init__(penalty=penalty, tol=tol, C=C, solver=solver)

    def fit(self, X, y):
        X_trans, y_trans = pairwise(X, y)
        fit_ = super(RankSVML, self).fit(X_trans, y_trans)
        return fit_


    def predict(self, X):
        pred = super(RankSVML, self).predict(X)
        return pred


    def predict_proba(self, X):
        pred = super(RankSVML, self).predict_proba(X)
        return pred

    def score(self, X, y):
        X_trans, y_trans = pairwise(X, y)
        return np.mean(super(RankSVML, self).predict(X_trans) == y_trans)



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

