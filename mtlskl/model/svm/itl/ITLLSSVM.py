
from mtlskl.model.svm.itl.ITLKernelModel import ITLKernelModel
from sklearn.metrics import accuracy_score, r2_score
from kernel.LSSVM import LSSVC, LSSVR
import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import label_binarize

class ITLLSSVC(ITLKernelModel):
    """docstring for ITLLSSVC."""

    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', random_state=None, task_info=-1):
        """Initialize the ITLLSSVC class."""
        super(ITLLSSVC, self).__init__(C, kernel, degree, gamma,
                                     coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, task_info)
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.score_fun = accuracy_score

    def fit(self, X, y, sample_weight=None):
        """Fit the ITLLSSVR."""
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVC(self.C, kernel, self.coef0,
                           self.degree, gamma)
        super(ITLLSSVC, self).fit(X, y, sample_weight)

    def predict(self, X):
        """Predicts using the trained ITLKernelModel."""
        n, m = X.shape

        K = rbf_kernel(X, self.X_train, gamma=self.gamma)

        pred = self.estim.predict(K)
        
        pred_bin = label_binarize(pred, classes=np.unique(pred))
        # ic(pred_bin[:4])

        return pred_bin

    def decision_function(self, X):
        """Return the decision function for the SVC."""
        n, m = X.shape
        task_col = self.task_info
        data_columns = np.delete(range(m), task_col)
        X_data = X[:, data_columns]
        return self.estim.decision_function(X_data)


class ITLLSSVR(ITLKernelModel):
    """docstring for ITLLSSVM."""

    def __init__(self, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, task_info=-1):
        """Initialize ITLLSSVR object."""
        super(ITLLSSVR, self).__init__(C, kernel, degree, gamma,
                                     coef0, shrinking, tol,
                                     cache_size,
                                     verbose, max_iter, task_info)
        self.epsilon = epsilon
        self.score_fun = r2_score

    def fit(self, X, y, sample_weight=None):
        """Fit the ITLLSSVR."""
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.estim = LSSVR(self.C, kernel, self.coef0,
                           self.degree, gamma)
        super(ITLLSSVR, self).fit(X, y, sample_weight)