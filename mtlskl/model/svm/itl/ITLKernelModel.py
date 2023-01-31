"""Class that implements a SingleTaskLearning SVM."""
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator
import numpy as np
import numpy_indexed as npi
from sklearn.base import clone

from icecream import ic

from sklearn.metrics.pairwise import rbf_kernel

class ITLKernelModel(BaseEstimator):
    """docstring for ITLKernelModel."""

    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, task_info=None):
        """Init for ITLKernelModel."""
        super(ITLKernelModel, self).__init__()
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
    
    def fit(self, X, y, task_info, sample_weight=None):
        """Fits the ITLKernelModel."""
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


        # groups
        self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        groups_dic = dict(zip(self.unique, self.groups_idx))

        self.estim_dic = {}

        for t, t_idx in groups_dic.items():
            self.estim_dic[t] = clone(self.estim)
                        
            if isinstance(self.gamma, float):
                gamma_t = self.gamma
            elif isinstance(self.gamma, dict):
                if t not in self.gamma:
                    raise AttributeError('{} is not a key of the gamma dictionaries')
                gamma_t = self.gamma[t]

            if sample_weight is None:
                sample_weight_t = None
            else:
                sample_weight_t = sample_weight[t_idx]

            X_data = X[t_idx[:, None], data_columns]
            K_t = self._get_kernel_train(X_data, X_data, gamma=gamma_t)
            y_t = y[t_idx]
        
            self.estim_dic[t].fit(K_t, y_t, sample_weight=sample_weight_t)

        # self.support_ = self.estim.support_
        # self.support_vectors_ = self.estim.support_vectors_
        # self.dual_coef_ = self.estim.dual_coef_
        # self.intercept_ = self.estim.intercept_
        return self

    def predict(self, X):
        """Predicts using the trained ITLKernelModel."""
        n, m = X.shape

        task_col = self.task_info
        data_columns = np.delete(range(m), task_col)

        # groups
        unique, groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        groups_dic = dict(zip(self.unique, self.groups_idx))
        groups_dic_train = dict(zip(self.unique, self.groups_idx))

        pred = np.zeros(X.shape[0])

        for t, t_idx in groups_dic.items():
            if t not in self.unique:
                raise AttributeError('Task {} was not in the training set')

            if isinstance(self.gamma, float):
                gamma_t = self.gamma
            elif isinstance(self.gamma, dict):
                if t not in self.gamma:
                    raise AttributeError('{} is not a key of the gamma dictionaries')
                gamma_t = self.gamma[t]
            
            t_idx_train = groups_dic_train[t]
            X_train_data = self.X_train[t_idx_train[:, None], data_columns]
            X_data = X[t_idx[:, None], data_columns]
            K_t = rbf_kernel(X_data, X_train_data, gamma=gamma_t)

            pred[t_idx] = self.estim_dic[t].predict(K_t)
        
        return pred

    def score(self, X, y, sample_weight=None, scoring=None):
        """Return score of ITLKernelModel."""
        pred = self.predict(X)
        if scoring is None:
            score = self.score_fun(y, pred)
        else:
            score = scoring(y, pred)
        return score

