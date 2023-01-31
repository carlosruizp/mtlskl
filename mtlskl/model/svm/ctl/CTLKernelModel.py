"""Class that implements a SingleTaskLearning SVM."""
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator
import numpy as np
import numpy_indexed as npi
from copy import deepcopy

from icecream import ic

from sklearn.metrics.pairwise import rbf_kernel


class CTLKernelModel(BaseEstimator):
    """docstring for CTLKernelModel."""

    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, task_info=None):
        """Init for CTLKernelModel."""
        super(CTLKernelModel, self).__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.estim_arr = []
        self.task_info = task_info

    
    def _get_kernel(self, X, Y, gamma):
        return rbf_kernel(X, Y, gamma=gamma)

    def _get_kernel_train(self, X, Y, gamma):
        return rbf_kernel(X, Y, gamma=gamma)
    
    def fit(self, X, y, task_info=None, sample_weight=None):
        """Fits the CTLKernelModel."""
        n, m = X.shape
        if self.gamma == 'auto':
            self.gamma = 1. / (m-1)
        if self.task_info is None:
            self.task_info = task_info
        task_col = self.task_info
        data_columns = np.delete(range(m), task_col)
        self.X_train = X
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self.y = y_2d

        X_data = X[:, data_columns]

        K = self._get_kernel_train(X_data, X_data, gamma=self.gamma)
        
        self.estim.fit(K, y, sample_weight=sample_weight)
        # #Precomputed Kernel
        # G_train = rbf_kernel(X[:, data_columns], X[:, data_columns], self.gamma)
        # self.estim.kernel = 'precomputed'
        # self.estim.fit(G_train, y)

        self.support_ = self.estim.support_
        self.support_vectors_ = self.estim.support_vectors_
        self.dual_coef_ = self.estim.dual_coef_
        self.intercept_ = self.estim.intercept_
        return self

    def predict(self, X):
        """Predicts using the trained CTLKernelModel."""
        n, m = X.shape
        task_col = self.task_info
        data_columns = np.delete(range(m), task_col)
        unique, groups_idx = npi.group_by(X[:, task_col],
                                          np.arange(n))
        X_data = X[:, data_columns]
        X_train_data = self.X_train[:, data_columns]

        K = rbf_kernel(X_data, X_train_data, gamma=self.gamma)

        pred = self.estim.predict(K)

        return pred

    def score(self, X, y, sample_weight=None, scoring=None):
        """Return score of CTLKernelModel."""
        pred = self.predict(X)
        if scoring is None:
            score = self.score_fun(y, pred)
        else:
            score = scoring(y, pred)
        return score

