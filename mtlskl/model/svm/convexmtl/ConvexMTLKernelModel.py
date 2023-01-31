from sklearn.svm import SVC
from sklearn.svm import SVR
import sklearn.metrics.pairwise as pairwise
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator
import numpy as np
import numpy_indexed as npi
import types

from icecream import ic

def neg_r2_score(y_true, y_pred, sample_weight=None):
    return -r2_score(y_true, y_pred, sample_weight)


class ConvexMTLKernelModel(BaseEstimator):

    """docstring for ConvexMTLKernelModel."""
    def __init__(self, C=1.0, ckernel='rbf', skernel='rbf', degree=3,
                 cgamma='auto', sgamma='auto', coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, lamb=1.0, task_info=None):
        super(ConvexMTLKernelModel, self).__init__()
        self.C = C
        self.ckernel = ckernel
        self.skernel = skernel
        self.degree = degree
        self.cgamma = cgamma
        self.sgamma = sgamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.lamb = lamb
        self.task_info = task_info
        if isinstance(lamb, dict) or isinstance(lamb, list):
            self.sparam=True # Pasar como parámetro?
        else:
            self.sparam = False

    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        self.X_train = X
        n, m = X.shape

        if self.task_info is None:
            self.task_info = task_info
        task_col = self.task_info
        self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        
        if self.cgamma == 'auto':
            self.cgamma_ = 1. / (m-1)
        else:
            self.cgamma_ = self.cgamma
        if self.sgamma == 'auto':
            self.sgamma_ = self.cgamma_
        else:
            self.sgamma_ = self.sgamma

        
        G_train = self._mtl_kernel_train(X, X, self.ckernel, self.skernel,
                                   task_info, self.lamb)

        # ic(G_train)
        # if y.ndim == 1:
        #     y_2d = y.reshape(-1, 1)
        # else:
        #     y_2d = y
        self.y = y
        self.estim.fit(G_train, self.y, sample_weight)
        # print(G_train)
        # print(self.estim)
        self.support_ = self.estim.support_
        self.support_vectors_ = self.estim.support_vectors_
        self.dual_coef_ = self.estim.dual_coef_
        # self.coef_ = self.estim.coef_
        self.intercept_ = self.estim.intercept_
        self.sample_weight = sample_weight
        self.task_info = task_info
        return self

    def predict(self, X):
        G_test = self._mtl_kernel(X, self.X_train, self.ckernel, self.skernel,
                                  self.task_info, self.lamb, cgamma=self.cgamma_,
                                  sgamma=self.sgamma_)
        return self.estim.predict(G_test)

    # def predict_common(self, X):
    #     G_test = self._ctl_kernel(X, self.X_train, self.ckernel, self.skernel, self.task_info, {}, cgamma=self.cgamma,
    #                                sgamma=self.sgamma)
    #     return self.estim.predict(G_test)

    # def predict_task(self, X, task):
    #     assert task in self.unique
    #     n = X.shape[0]
    #     t_col = np.array([task] * n)[:, None]
    #     task_col = self.task_info
    #     X_data = np.delete(X, task_col, axis=1).astype(float)
    #     X2task = np.concatenate((X_data, t_col), axis=1)
    #     G_test = self._itl_kernel(X2task, self.X_train, self.ckernel, self.skernel, self.task_info, {}, cgamma=self.cgamma,
    #                                sgamma=self.sgamma)
    #     # print(G_test)
    #     return self.estim.predict(G_test)

    def score(self, X, y, sample_weight=None, scoring=None):
        G_test = self._mtl_kernel(X, self.X_train, self.ckernel, self.skernel,
                                  self.task_info, self.lamb, cgamma=self.cgamma_,
                                  sgamma=self.sgamma_)
        y_pred = self.estim.predict(G_test)

        n, m = X.shape
        task_col = self.task_info
        unique, groups_idx = npi.group_by(X[:, task_col],
                                          np.arange(n))
        self.scores = {}
        for i, t in enumerate(unique):
            y_true_g = y[groups_idx[i]]
            y_pred_g = y_pred[groups_idx[i]]
            if scoring is None:
                self.scores[t] = self.score_fun(y_true_g, y_pred_g)
            else:
                self.scores[t] = scoring(y_true_g, y_pred_g)

        if scoring is None:
            return self.score_fun(y, y_pred, sample_weight)
        else:
            return scoring(y, y_pred, sample_weight)

    def _get_kernel_fun(self, kernel):
        if isinstance(kernel, str):
            kernel_f = getattr(pairwise, kernel+'_kernel')
        else:
            kernel_f = kernel
        return kernel_f

    def _apply_kernel(self, kernel, x, y, **kwargs):
        kernel_f = self._get_kernel_fun(kernel)
        if kernel_f == pairwise.rbf_kernel:
            if 'gamma' not in kwargs:
                if kwargs['common']:
                    gamma = kwargs['cgamma']
                else:
                    if 'task' not in kwargs:
                        gamma = kwargs['sgamma']
                    else:
                        if kwargs['task'] in kwargs['sgamma']:
                            gamma = kwargs['sgamma'][kwargs['task']]
                        else:
                            gamma = kwargs['sgamma'][float(kwargs['task'])]
            else:
                gamma = kwargs['gamma']
            return kernel_f(x, y, gamma)
        else:
            return kernel_f(x, y)

    def _compute_K(self, x, y, tx, ty, skernel, **kwargs):
        skernel_isList = isinstance(skernel, (list, np.ndarray))
        skernel_isDic = isinstance(skernel, dict)
        if tx == ty:
            if skernel_isList:
                itx = np.where(self.unique == tx)[0][0]
                ret = self._apply_kernel(skernel[itx], x, y,
                                         **dict(kwargs, common=False,
                                                task=itx))
            elif skernel_isDic:
                ret = self._apply_kernel(skernel[tx], x, y,
                                         **dict(kwargs, common=False))
            else:
                sgamma_isDic = isinstance(kwargs['sgamma'], dict)
                if sgamma_isDic:
                    ret = self._apply_kernel(skernel, x, y,
                                            **dict(kwargs, common=False, task=tx))
                else:
                    ret = self._apply_kernel(skernel, x, y,
                                    **dict(kwargs, common=False))
        else:
            ret = 0
        return ret

    def _compute_Q(self, x, y, tx, ty, ckernel, **kwargs):
        return self._apply_kernel(ckernel, x, y, **dict(kwargs, common=True))

    def _mtl_kernel(self, X, Y, ckernel, skernel, task_info=0, mu=1, **kwargs):
        return self._mtl_kernel_convex(X, Y, ckernel, skernel, task_info,
                                                cgamma=self.cgamma_, sgamma=self.sgamma_)

    def _mtl_kernel_train(self, X, Y, ckernel, skernel, task_info=0, mu=1, **kwargs):        
        return self._mtl_kernel_convex(X, Y, ckernel, skernel, task_info,
                                                cgamma=self.cgamma_, sgamma=self.sgamma_)

    
    def _mtl_kernel_convex(self, X, Y, ckernel, skernel, task_info, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        K = np.zeros((nX, nY))
        Q = np.zeros((nX, nY))

        for i, tx in enumerate(unique_X):
            if self.sparam:
                if isinstance(tx, float):
                    tx = int(tx)
                lambX = self.lamb[str(tx)]
            else:
                lambX = self.lamb
            for j, ty in enumerate(unique_Y):
                if self.sparam:
                    if isinstance(ty, float):
                        ty = int(ty)
                    lambY = self.lamb[str(ty)]
                else:
                    lambY = self.lamb
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                Q[indX[:, None], indY] = lambX * lambY * self._compute_Q(X_data[indX],
                                                                         Y_data[indY],
                                                                         tx, ty,
                                                                         ckernel, **kwargs)
                if tx == ty:
                    K[indX[:, None], indY] = (1-lambX)*(1-lambY)*self._compute_K(X_data[indX],
                                                                                 Y_data[indY],
                                                                                 tx, ty,
                                                                                 skernel, **kwargs)
        # ic(Q)
        # ic(K)
        # ic(np.sum(K, axis=0))
        hat_Q = Q + K
        return hat_Q

