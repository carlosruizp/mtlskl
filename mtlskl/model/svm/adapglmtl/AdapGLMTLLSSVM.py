from sklearn.svm import SVC
from sklearn.svm import SVR
import sklearn.metrics.pairwise as pairwise
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import hinge_loss
from sklearn.base import BaseEstimator
import numpy as np
import numpy_indexed as npi
import types
import pickle
from notebooks.utils import get_train, timer, profile
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy

from sklearn.metrics.pairwise import pairwise_distances
from icecream import ic


from kernel.LSSVM import LSSVC, LSSVR
import sklearn.metrics.pairwise as pairwise
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import hinge_loss
from sklearn.base import BaseEstimator
import numpy as np
import numpy_indexed as npi
import types
import pickle
from notebooks.utils import get_train, timer, profile
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy

from icecream import ic

from scipy.stats import entropy

from kernel.adapGraphLap.utils import *
from kernel.adapGraphLap.AdapGLKernelModel import AdapGLKernelModel



class AdapGLMTLLSSVM(AdapGLKernelModel):
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', cgamma=None, sgamma=None, coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, nu=1.0, task_info=None,
                 max_iter_ext=-1, opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0):
        super(AdapGLMTLLSSVM, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                                     cgamma=cgamma, sgamma=sgamma,
                                     coef0=coef0,
                                     shrinking=shrinking, tol=tol, cache_size=cache_size, verbose=verbose,
                                     max_iter=max_iter, nu=nu, task_info=task_info, max_iter_ext=max_iter_ext,
                                     opt=opt, tol_ext=tol_ext, delta=delta, order_delta=order_delta, mu=mu, ind_reg=ind_reg, lamb=lamb)


class AdapGLMTLLSSVC(AdapGLMTLLSSVM):
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', cgamma=None, sgamma=None, coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, task_info=None,
                 decision_function_shape='ovo', random_state=None, nu=1.0, max_iter_ext=-1,
                 opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0):
        super(AdapGLMTLLSSVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                                     cgamma=cgamma, sgamma=sgamma,
                                     coef0=coef0,
                                     shrinking=shrinking, tol=tol, cache_size=cache_size, verbose=verbose,
                                     max_iter=max_iter, nu=nu, task_info=task_info, max_iter_ext=max_iter_ext,
                                     opt=opt, tol_ext=tol_ext, delta=delta, order_delta=order_delta, mu=mu, ind_reg=ind_reg, lamb=lamb)
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVC(self.C, kernel, self.coef0,
                           self.degree, gamma)
        return super().fit(X, y, task_info, **kwargs)
    
    def decision_function(self, X, G=None):
        if G is None:
            task_col = self.task_info
            X_train_data = np.delete(self.X_train, task_col, axis=1).astype(float)
            X_test_data = np.delete(X, task_col, axis=1).astype(float)

            G_test_standard = apply_kernel(self.kernel, X_test_data, X_train_data, self.gamma)
            G_test_fused = mtl_kernel_graphlap(X, self.X_train, self.kernel,
                                            gamma=self.gamma,
                                            deltainv=self.deltainv,
                                            order_delta=self.order_delta,
                                            task_info=self.task_info,
                                            G=G_test_standard)

            G_test = self.lamb**2 * G_test_standard + (1 - self.lamb)**2 * G_test_fused
        else:
            G_test = G
        return self.estim.decision_function(G_test)

    def score(self, X, y, G=None, sample_weight=None, scoring=None):
        pred = self.predict(X, G=G)
        errors = (y - pred)**2
        return np.mean(errors)


class AdapGLMTLLSSVR(AdapGLMTLLSSVM):
    """docstring for mtlSVM."""
    def __init__(self, kernel='rbf', degree=3, gamma='auto', cgamma=None, sgamma=None,
                 coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, nu=1.0, task_info=None, max_iter_ext=-1, opt='entropy',
                 tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0):
        super(AdapGLMTLLSSVM, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                                     cgamma=cgamma, sgamma=sgamma,
                                     coef0=coef0,
                                     shrinking=shrinking, tol=tol, cache_size=cache_size, verbose=verbose,
                                     max_iter=max_iter, nu=nu, task_info=task_info, max_iter_ext=max_iter_ext,
                                     opt=opt, tol_ext=tol_ext, delta=delta, order_delta=order_delta, mu=mu, ind_reg=ind_reg, lamb=lamb)
        self.epsilon = epsilon


    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVR(self.C, kernel, self.coef0,
                           self.degree, gamma)
        return super().fit(X, y, task_info, **kwargs)

    def score(self, X, y, G=None, sample_weight=None, scoring=None):
        pred = self.predict(X, G=G)
        errors = (y - pred)**2
        return np.mean(errors)

    def decision_function(self, X, G=None):
        return self.predict(X, G=G)
    