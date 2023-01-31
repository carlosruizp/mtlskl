from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from mtlskl.model.svm.convexmtl.ConvexMTLKernelModel import ConvexMTLKernelModel

from icecream import ic

class ConvexMTLSVC(ConvexMTLKernelModel):
    """docstring for ConvexMTLSVM."""
    def __init__(self, C=1.0, ckernel='rbf', skernel='rbf', degree=3,
                 cgamma='auto', sgamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, task_info=None,
                 decision_function_shape='ovr', random_state=None, lamb=1.0):
        super(ConvexMTLSVC, self).__init__(C, ckernel, skernel, degree, cgamma,
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
        self.estim = SVC(C=self.C, kernel=kernel, degree=self.degree, gamma=gamma, coef0=self.coef0,
                       shrinking=self.shrinking, probability=self.probability, tol=self.tol, cache_size=self.cache_size,
                       class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter,
                       decision_function_shape=self.decision_function_shape, random_state=self.random_state)
        return super().fit(X, y, task_info, **kwargs)

    def decision_function(self, X):
        return self.estim.decision_function(X)


class ConvexMTLSVR(ConvexMTLKernelModel):
    """docstring for ConvexMTLSVM."""
    def __init__(self, ckernel='rbf', skernel='rbf', degree=3, cgamma='auto',
                 sgamma='auto', coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, lamb=1.0, task_info=None):
        super(ConvexMTLSVR, self).__init__(C, ckernel, skernel, degree, cgamma,
                                     sgamma, coef0, shrinking, tol, cache_size,
                                     verbose, max_iter, lamb, task_info)
        self.epsilon = epsilon
        self.score_fun = r2_score

    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = SVR(kernel=kernel, degree=self.degree, gamma=gamma, coef0=self.coef0,
                       tol=self.tol, C=self.C, epsilon=self.epsilon, shrinking=self.shrinking, cache_size=self.cache_size,
                       verbose=self.verbose, max_iter=self.max_iter)
        return super().fit(X, y, task_info, **kwargs)
