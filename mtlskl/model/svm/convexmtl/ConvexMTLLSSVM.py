from sklearn.metrics import accuracy_score, r2_score
from kernel.LSSVM import LSSVC, LSSVR
import numpy as np
from kernel.convexmtl.ConvexMTLKernelModel import ConvexMTLKernelModel

class ConvexMTLLSSVC(ConvexMTLKernelModel):
    """docstring for ConvexMTLLSSVM."""
    def __init__(self, C=1.0, ckernel='rbf', skernel='rbf', degree=3,
                 cgamma='auto', sgamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, task_info=None,
                 decision_function_shape='ovr', random_state=None, lamb=1.0):
        super(ConvexMTLLSSVC, self).__init__(C, ckernel, skernel, degree, cgamma,
                                     sgamma, coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, lamb, task_info)
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.score_fun = accuracy_score

    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVC(self.C, kernel, self.coef0,
                           self.degree, gamma)
        return super().fit(X, y, task_info, **kwargs)

    def decision_function(self, X):
        return self.estim.decision_function(X)


class ConvexMTLLSSVR(ConvexMTLKernelModel):
    """docstring for ConvexMTLLSSVM."""
    def __init__(self, ckernel='rbf', skernel='rbf', degree=3, cgamma='auto',
                 sgamma='auto', coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, lamb=1.0, task_info=None):
        super(ConvexMTLLSSVR, self).__init__(C, ckernel, skernel, degree, cgamma,
                                     sgamma, coef0, shrinking, tol, cache_size,
                                     verbose, max_iter, lamb, task_info)
        self.epsilon = epsilon
        self.score_fun = r2_score

    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVR(self.C, kernel, self.coef0,
                           self.degree, gamma)
        return super().fit(X, y, task_info, **kwargs)
