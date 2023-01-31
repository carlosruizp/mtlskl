
from mtlskl.model.svm.ctl.CTLKernelModel import CTLKernelModel
from sklearn.metrics import accuracy_score, r2_score
from kernel.LSSVM import LSSVC, LSSVR
import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import label_binarize

class CTLLSSVC(CTLKernelModel):
    """docstring for CTLLSSVC."""

    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', random_state=None, task_info=-1):
        """Initialize the CTLLSSVC class."""
        super(CTLLSSVC, self).__init__(C, kernel, degree, gamma,
                                     coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, task_info)
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.score_fun = accuracy_score

    def fit(self, X, y, task_info=-1, sample_weight=None):
        """Fit the CTLLSSVR."""
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVC(self.C, kernel, self.coef0,
                           self.degree, gamma)
        super(CTLLSSVC, self).fit(X, y, task_info, sample_weight)

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

        pred_bin = label_binarize(pred, classes=np.unique(pred))

        return pred_bin

    def decision_function(self, X):
        """Return the decision function for the SVC."""
        n, m = X.shape
        task_col = self.task_info
        data_columns = np.delete(range(m), task_col)
        X_data = X[:, data_columns]
        return self.estim.decision_function(X_data)


class CTLLSSVR(CTLKernelModel):
    """docstring for CTLLSSVM."""

    def __init__(self, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, task_info=-1):
        """Initialize CTLLSSVR object."""
        super(CTLLSSVR, self).__init__(C, kernel, degree, gamma,
                                     coef0, shrinking, tol,
                                     cache_size,
                                     verbose, max_iter, task_info)
        self.epsilon = epsilon
        self.score_fun = r2_score

    def fit(self, X, y, task_info=-1, sample_weight=None):
        """Fit the CTLLSSVR."""
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVR(self.C, kernel, self.coef0,
                           self.degree, gamma)
        super(CTLLSSVR, self).fit(X, y, task_info, sample_weight)