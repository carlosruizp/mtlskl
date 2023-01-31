from mtlskl.model.svm.ctl.CTLKernelModel import CTLKernelModel
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.svm import SVR
import numpy as np

class CTLSVC(CTLKernelModel):
    """docstring for CTLSVC."""

    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', random_state=None, task_info=-1):
        """Initialize the CTLSVC class."""
        super(CTLSVC, self).__init__(C, kernel, degree, gamma,
                                     coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, task_info)
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.score_fun = accuracy_score

    def fit(self, X, y, task_info=-1, sample_weight=None):
        """Fit the CTLSVR."""
        self.estim = SVC(C=self.C, kernel='precomputed', degree=self.degree, gamma='auto',
                        coef0=self.coef0,
                        shrinking=self.shrinking, probability=self.probability, tol=self.tol,
                        cache_size=self.cache_size,
                        class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter,
                        decision_function_shape=self.decision_function_shape, random_state=self.random_state)
        return super(CTLSVC, self).fit(X, y, task_info, sample_weight)

    def decision_function(self, X):
        """Return the decision function for the SVC."""
        n, m = X.shape
        task_col = self.task_info
        data_columns = np.delete(range(m), task_col)
        X_data = X[:, data_columns]
        return self.estim.decision_function(X_data)


class CTLSVR(CTLKernelModel):
    """docstring for CTLSVM."""

    def __init__(self, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, task_info=-1):
        """Initialize CTLSVR object."""
        super(CTLSVR, self).__init__(C, kernel, degree, gamma,
                                     coef0, shrinking, tol,
                                     cache_size,
                                     verbose, max_iter, task_info)
        self.epsilon = epsilon
        self.score_fun = r2_score

    def fit(self, X, y, task_info=-1, sample_weight=None):
        """Fit the CTLSVR."""
        self.estim = SVR(kernel='precomputed', degree=self.degree, gamma='auto', coef0=self.coef0,
                       tol=self.tol, C=self.C, epsilon=self.epsilon, shrinking=self.shrinking,
                       cache_size=self.cache_size,
                       verbose=self.verbose, max_iter=self.max_iter)
        return super(CTLSVR, self).fit(X, y, task_info, sample_weight)