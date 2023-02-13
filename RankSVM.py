# Implementation of RankSVM 
# Uses classifiers for scikit-learn: LinearSVC, LinearSVR and LogisticRegression
# author: Myrto Limnios // mail: myli@math.ku.dk


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


