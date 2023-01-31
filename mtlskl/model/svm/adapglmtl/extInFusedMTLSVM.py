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

from scipy.stats import entropy


aux_dir = 'aux_files'

def rbfkernel(dists, gamma):
    return np.exp(-(gamma)*dists)

def is_pos_def(x):
    # print('eigs:')
    # print(np.linalg.eigvals(x))
    return np.all(np.linalg.eigvals(x) > 0)

def is_pos_semidef(x, tol=1e-16):
    # print('eigs:')
    # print(np.linalg.eigvals(x))
    return np.all(np.linalg.eigvals(x) >= 0 - tol)


def epsilon_insensitive_error(ytrue, ypred, epsilon):

    ytrue_flat = ytrue.flatten()
    ypred_flat = ypred.flatten()
    if ytrue_flat.shape != ypred_flat.shape:
        raise Exception('True value and prediction have not equal lengths')

    err = np.abs(ytrue_flat - ypred_flat) - epsilon    
    return np.sum(np.clip(err, 0, None))


def rows_entropy(M):
    s = np.sum([entropy(row) for row in M])
    return s


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
        

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
       

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw )
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=10)
    cbar.ax.tick_params(labelsize=18)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(fontsize=10)
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def remove_diag(matrix):
    m = matrix.copy()
    for i, r in enumerate(matrix):
        m[i, i] = 0
    return m

def normalize_dists(dists, dots):
    dists_norm = np.zeros(dists.shape)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            dists_norm[i, j] = dists[i, j] / (dots[i, i] + dots[j, j])
    return dists_norm


class extInFusedMTLSVM(BaseEstimator):
    'init'
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, nu=1.0, task_info=None,
                 max_iter_ext=-1, opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0, inner_kernel=True):
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
        self.max_iter_ext = max_iter_ext
        self.tol = tol
        self.task_info = task_info
        self.nu = nu # 1e-3
        self.opt = opt
        self.tol_ext = tol_ext
        self.delta = delta
        self.order_delta = order_delta
        self.mu = mu
        self.ind_reg = ind_reg
        self.lamb = lamb
        self.inner_kernel = inner_kernel
        
    # def fit_gsearch(self, X, y, task_info=-1, sample_weight=None, **kwargs):
    #     self.X_train = X
    #     self.y_train = y
    #     n, m = X.shape
    #     if self.task_info is None:
    #         self.task_info = task_info
    #     task_col = self.task_info
    #     self.unique, self.groups_idx = npi.group_by(X[:, task_col],
    #                                                 np.arange(n))

    #     if hasattr(self, 'epsilon'):
    #         name_comb = '{}/{:.12f}_{:.12f}_{:.12f}'.format(aux_dir, self.C, self.epsilon, self.nu)
    #     else:
    #         name_comb = '{}/{:.12f}_{:.12f}'.format(aux_dir, self.C, self.nu)

    #     if 'order_delta' in kwargs:
    #         self.order_delta = kwargs['order_delta']
    #     if self.gamma == 'auto':
    #         self.gamma = 1
    #     else:
    #         self.gamma = self.gamma
    #     if self.gamma == 'auto':
    #         self.gamma = self.gamma
    #     else:
    #         self.gamma = self.gamma
        
    #     if self.max_iter_ext == 1:
    #         if 'delta' in kwargs:
    #             self.delta = kwargs['delta']
    #         else:
    #             T = len(self.unique)
    #             delta = np.ones((T, T))
    #             for i in range(T):
    #                 delta[i, i] = 0
    #                 delta[i, :] /= -np.sum(delta[i, :])
    #                 delta[i, i] = -np.sum(delta[i, :])
    #             self.delta = delta / np.sum(np.abs(delta))

    #         self.deltainv_hist_ = []
    #         self.delta_hist_ = []
    #         self.dual_coef_hist_ = []
    #         self.support_hist_ = []
    #         self.intercept_hist_ = []
    #         self.dots_hist_ = []
    #         self.dists_hist_ = []

    #     else:
    #         # print('ITER > {} ---------------------------------------------------------'.format(self.max_iter_ext))  
    #         # print(name_comb)
    #         with open('{}_iter{}_self.p'.format(name_comb, self.max_iter_ext-1), 'rb') as file:
    #             prev_model = pickle.load(file)
    #             self.deltainv_hist_ = prev_model.deltainv_hist_
    #             self.delta_hist_ = prev_model.delta_hist_
    #             self.dual_coef_hist_ = prev_model.dual_coef_hist_
    #             self.support_hist_ = prev_model.support_hist_
    #             self.intercept_hist_ = prev_model.intercept_hist_
    #             self.dots_hist_ = prev_model.dots_hist_
    #             self.dists_hist_ = prev_model.dists_hist_
    #             self.delta = prev_model.delta
    #             self.G_train = prev_model.G_train

    #     # Delta Fixed and Optimize SVM       
    #     G_train = self._mtl_kernel_fused(X, X, self.kernel, self.kernel,
    #                                      task_info, G_train_standard)
    #     self.y = y

    #     # print(G_train)
        
    #     self.svm.fit(G_train, self.y, sample_weight)
    #     self.support_ = self.svm.support_
    #     self.support_vectors_ = self.svm.support_vectors_
    #     self.dual_coef_ = self.svm.dual_coef_
    #     # self.coef_ = self.svm.coef_
    #     self.intercept_ = self.svm.intercept_
    #     self.sample_weight = sample_weight
    #     self.task_info = task_info

    #     self.deltainv_hist_.append(self.deltainv)
    #     self.delta_hist_.append(self.delta)
    #     self.dual_coef_hist_.append(self.dual_coef_)
    #     self.support_hist_.append(self.support_)
    #     self.intercept_hist_.append(self.intercept_)

    #     # W fixed and optimize Delta
    #     self.delta_old = self.delta
    #     self.optimize_laplacian(G_train)
    #     self.dots_hist_.append(self.dots.copy())
    #     self.dists_hist_.append(self.dists.copy())

        
    #     with open('{}_iter{}_self.p'.format(name_comb, self.max_iter_ext), 'wb') as file:
    #         pickle.dump(self, file)

    
    
    def compute_objf(self, X, y, G_train_standard, dists, dots):
        # print('MATRICES')
        # print(self.B)
        # print(self.laplacian)
        # print(self.delta)
        # print(self.delta_old)

        # reg
        alpha = self.svm._dual_coef_.T
        reg_common = self.lamb**2 * (alpha.T @ G_train_standard[self.svm.support_, self.svm.support_[:, None]] @ alpha)[0][0]
        G_train_graphL = self._mtl_kernel_fused(X, X, self.kernel, self.gamma, self.delta, 
                                            self.task_info, G_train_standard)
        G_train = (1-self.lamb)**2 * G_train_graphL
        
        # reg = (alpha.T @ G_train[self.svm.support_, self.svm.support_[:, None]] @ alpha)[0][0]         
        # print('Reg')
        # print(reg)


        reg = (self.nu * np.sum(self.B * dists) + np.trace(dots)) # (1-self.lamb)**2  está en dots
        # print(reg)

        reg = reg + reg_common

        self.reg_hist_.append(reg)
        
        # entropy
        ent = rows_entropy(self.B)
        self.ent_hist_.append(ent)
        # print('entropy:')
        # print(ent)
            
        # print('Prueba')
        # print(self.lamb)
        # print(-self.mu *ent + self.nu * np.sum(self.B * dists) )

        # Score
        G_train_graphL = self._mtl_kernel_fused(X, X, self.kernel, self.gamma, self.delta_old, 
                                            self.task_info, G_train_standard)
        G_train = self.lamb**2 * G_train_standard + (1-self.lamb)**2 * G_train_graphL
        # print(alpha.shape)
        # print(G_train[:, self.svm.support_].shape)
        # pred = G_train[:, self.svm.support_] @ alpha + self.intercept_
        # pred = pred.flatten()

        score = self.score(X, y, G=G_train)
        self.score_hist_.append(score)

        objf = self.C * score + (1./2) * reg - (self.mu / 2) * ent

        # print('Score:', self.C * score)
        # print('Reg:',  reg )
        # print('Ent:', self.mu * ent)
        # print('Score + Reg:', self.C * score + reg)
        # print('Reg - Entropy:', reg - self.mu * ent)
        # print('Objf: ', objf)
        
        self.objf_hist_.append(objf)

        return objf
    
    @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def fit_no_gsearch(self, X, y, task_info=-1, sample_weight=None, **kwargs):

        self.nuaux = 1

        self.X_train = X
        self.y_train = y
        n, m = X.shape
        if self.task_info is None:
            self.task_info = task_info
        task_col = self.task_info
        self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        X_data = np.delete(X, task_col, axis=1).astype(float)

        if self.gamma == 'auto':
            self.gamma = 1
        else:
            self.gamma = self.gamma
        if self.gamma == 'auto':
            self.gamma = self.gamma
        else:
            self.gamma = self.gamma
        
        if 'delta' in kwargs:
            self.delta = kwargs['delta']
        else:
            T = len(self.unique)
            B = np.ones((T, T))
            B = B/T
            # B = np.identity(T) # Prueba con la identidad
            L = deepcopy(-B)
            for i in range(T):
                L[i, i] = np.sum(B[i, :]) - B[i, i]
            # B = B / np.sum(np.abs(B))
            self.B = B
            self.laplacian = L
            self.delta = self.laplacian + self.laplacian.T

        if 'order_delta' in kwargs:
            self.order_delta = kwargs['order_delta']
        else:
            self.order_delta = dict(zip(self.unique, range(len(self.unique))))

        self.deltainv_hist_ = []
        self.delta_hist_ = []
        self.laplacian_hist_ = []
        self.B_hist_ = []
        self.dual_coef_hist_ = []
        self.support_hist_ = []
        self.intercept_hist_ = []
        self.dots_hist_ = []
        self.dists_hist_ = []
        self.score_hist_ = []
        self.reg_hist_ = []
        self.ent_hist_ = []
        self.objf_hist_ = []
        score = 0
        # print('FIT')
        # Get G_train
        G_train_standard = self._apply_kernel(self.kernel, X_data, X_data, self.gamma)

        # print(self.max_iter_ext)
        iter = 0
        stopCondition = False
        objf = 0
        
        while not stopCondition:
            # print('Delta')
            # print(self.delta)
            # Delta Fixed and Optimize SVM       
            G_train_graphL = self._mtl_kernel_fused(X, X, self.kernel, self.gamma, self.delta,
                                             task_info, G_train_standard)

            # print('G_graphL')
            # print(G_train_graphL)
            
            # print(self.deltainv)
            self.y = y
            
            G_train = self.lamb**2 * G_train_standard + (1-self.lamb)**2 * G_train_graphL
                

            self.svm.fit(G_train, self.y, sample_weight)
            self.support_ = self.svm.support_
            self.support_vectors_ = self.svm.support_vectors_
            self.dual_coef_ = self.svm.dual_coef_
            
            self.intercept_ = self.svm.intercept_
            self.sample_weight = sample_weight
            self.task_info = task_info

            self.deltainv_hist_.append(self.deltainv)
            self.delta_hist_.append(self.delta)

            self.B_hist_.append(self.B)
            self.laplacian_hist_.append(self.laplacian)
            
            self.dual_coef_hist_.append(self.dual_coef_)
            self.support_hist_.append(self.support_)
            self.intercept_hist_.append(self.intercept_)
            
            prev_objf = objf
            if (iter < self.max_iter_ext) or self.max_iter_ext < 0:
                # print('Fitted SVM')
                alpha = self.svm._dual_coef_.T
                dists, dots = self.distances_between_tasks(G_train_standard)
                self.delta_old = self.delta
                objf = self.compute_objf(X, y, G_train_standard, dists, dots)

            # print('Update L')
            

            #Stopping Condition
            stopCondition = np.abs(prev_objf - objf) < self.tol_ext
            stopCondition = stopCondition or ((iter >= self.max_iter_ext) if self.max_iter_ext > -1 else False)
            if not stopCondition:
                prev_objf = objf
                # W fixed and optimize Delta
                # self.delta_old = self.delta
                # print('Fitted SVM')
                alpha = self.svm._dual_coef_.T
                dists, dots = self.distances_between_tasks(G_train_standard)
                self.delta_old = self.delta
                objf = self.compute_objf(X, y, G_train_standard, dists, dots)
                self.optimize_laplacian(G_train_standard, dists, dots)
                self.dots_hist_.append(self.dots)
                self.dists_hist_.append(self.dists)
                objf = self.compute_objf(X, y, G_train_standard, dists, dots)
            else:
                break
            iter+=1
        self.total_iter = iter
            # print((iter >= self.max_iter_ext) if self.max_iter_ext > 0 else False)
    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        return self.fit_no_gsearch(X, y, task_info, sample_weight, **kwargs)


    def predict(self, X, G=None):
        if G is None:
            task_col = self.task_info
            X_train_data = np.delete(self.X_train, task_col, axis=1).astype(float)
            X_test_data = np.delete(X, task_col, axis=1).astype(float)

            G_test_standard = self._apply_kernel(self.kernel, X_test_data, X_train_data, self.gamma)
            G_test_fused = self._mtl_kernel_fused(X, self.X_train, self.kernel, self.gamma, self.delta, self.task_info)

            G_test = self.lamb**2 * G_test_standard + (1 - self.lamb)**2 * G_test_fused
        else:
            G_test = G

        # print(G_test)

        # pred_svm = self.svm.predict(G_test)
        # alpha = self.svm.dual_coef_.T
        # sv = self.svm.support_
        # print(alpha.flatten())
        # print(self.y_train[sv])
        # print(alpha.shape)
        
        # print(G_test.shape)
        # print(G_test[sv[:, None], sv].shape)
        # pred = G_test[:, sv] @ alpha + self.svm.intercept_

        # print(pred_svm)
        # print(np.sign(pred.flatten()))
        
        return self.svm.predict(G_test)

    def score(self, X, y, sample_weight=None, scoring=None):
        task_col = self.task_info
        X_train_data = np.delete(self.X_train, task_col, axis=1).astype(float)
        X_test_data = np.delete(X, task_col, axis=1).astype(float)

        G_test_standard = self._apply_kernel(self.kernel, X_test_data, X_train_data, self.gamma)
        G_test_fused = self._mtl_kernel_fused(X, self.X_train, self.kernel, self.gamma, self.delta, self.task_info)

        G_test = self.lamb**2 * G_test_standard + (1 - self.lamb)**2 * G_test_fused
        y_pred = self.svm.predict(G_test)

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
        # if not isinstance(kernel, (str, types.FunctionType)):
        #     raise Exception('kernel of wrong type')
        if isinstance(kernel, str):
            kernel_f = getattr(pairwise, kernel+'_kernel')
        else:
            kernel_f = kernel
        return kernel_f

    def _apply_kernel(self, kernel, x, y, gamma):
        kernel_f = self._get_kernel_fun(kernel)
        if kernel_f == pairwise.rbf_kernel:
            return kernel_f(x, y, gamma)
        else:
            return kernel_f(x, y)

    
    def compute_dists(self, X, Y, weighted=False, delta=None, task_info=-1):
        # X_data is an n_1 x d matrix
        # X_data is an n_2 x d matrix
        
        task_col = task_info
        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        dots = X_data @ Y_data.T
        dots_1 = X_data @ X_data.T
        dots_2 = Y_data @ Y_data.T
        if weighted:
            if self.ind_reg:
                delta_full = self.nu * delta + np.identity(delta.shape[0])
            else:
                delta_full = delta
  
            # print(is_pos_semidef(delta))
            self.deltainv = np.linalg.inv(delta_full)

            nX = X_data.shape[0]
            nY = Y_data.shape[0]
            task_X = X[:, task_col]
            task_Y = Y[:, task_col]
            unique_X, groups_idx_X = npi.group_by(task_X,
                                                np.arange(nX))
            unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                                np.arange(nY))
            order_delta = dict(zip(unique_X, range(len(unique_X))))
            A = np.zeros((nX, nY))
            for i, tx in enumerate(unique_X):
                for j, ty in enumerate(unique_Y):
                    indX = groups_idx_X[i]
                    indY = groups_idx_Y[j]
                    order_tx = order_delta[tx]
                    order_ty = order_delta[ty]
                    a_xy = self.deltainv[order_tx, order_ty]
                    A[indX[:, None], indY] = a_xy
        
            dots = np.multiply(A, dots)

        diag_1 = np.diagonal(dots_1)
        diag_2 = np.diagonal(dots_2)
        dists = -2 * dots + diag_2 + diag_1[:, None]
        return dists
    
    def mtl_kernel_fused_outer(self, X, Y, task_info=-1, delta=None):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info
        if delta is None:
            delta = self.delta

        if self.ind_reg:
            delta_full = self.nu * delta + np.identity(delta.shape[0])
        else:
            delta_full = delta

            
        # print(is_pos_semidef(delta))
        self.deltainv = np.linalg.inv(delta_full)
        
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
        order_delta = dict(zip(unique_X, range(len(unique_X))))
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                order_tx = order_delta[tx]
                order_ty = order_delta[ty]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy
        
        
        dists = self.compute_dists(X, Y, weighted=False)
        Q = rbfkernel(dists, self.gamma)

        Q_graphL = np.multiply(A, Q)
        
        return Q_graphL

    def mtl_kernel_fused_inner(self, X, Y, task_info=-1, delta=None):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info

        if delta is None:
            delta = self.delta

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        
        dists = self.compute_dists(X, Y, weighted=True, delta=delta)
        Q = rbfkernel(dists, self.gamma)

        Q_graphL = Q
        
        return Q_graphL
    

    def mtl_kernel_fused(self, X, Y, task_info=-1, delta=None):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info

        if delta is None:
            delta = self.delta

        if self.ind_reg:
            delta_full = self.nu * delta + np.identity(delta.shape[0])
        else:
            delta_full = delta

            
        # print(is_pos_semidef(delta))
        self.deltainv = np.linalg.inv(delta_full)

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
        order_delta = dict(zip(unique_X, range(len(unique_X))))
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                order_tx = order_delta[tx]
                order_ty = order_delta[ty]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy
        Q = self._apply_kernel(self.kernel, X_data, Y_data, self.gamma)
        Q_graphL = np.multiply(A, Q)
        
        return Q_graphL

    # @timer
    def _mtl_kernel_fused(self, X, Y, kernel, gamma, delta, task_info=-1, G_train=None):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        if self.inner_kernel:
            return self.mtl_kernel_fused_inner(X, Y, task_info=task_info, delta=delta)
        else:
            return self.mtl_kernel_fused_outer(X, Y, task_info=task_info, delta=delta)
        
    # @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    # def _matrix_dots(self, R, S):
    #     delta_inv = self.deltainv
    #     # print('dots')
    #     # print(delta_inv)
    #     X = Y = self.X_train
    #     task_col = self.task_info
    #     order_delta = self.order_delta
    #     nX = X.shape[0]
    #     nY = Y.shape[0]
    #     task_X = X[:, task_col]
    #     task_Y = Y[:, task_col]
    #     unique_X, groups_idx_X = npi.group_by(task_X,
    #                                               np.arange(nX))
    #     unique_Y, groups_idx_Y = npi.group_by(task_Y,
    #                                           np.arange(nY))
    #     M = np.zeros((nX, nY))    

    #     order_R = order_delta[R]
    #     order_S = order_delta[S]
    #     for i, tx in enumerate(unique_X):
    #         indX = groups_idx_X[i]
    #         order_tx = order_delta[tx]
    #         # order_R = order_delta[float(R)]
            
    #         for j, ty in enumerate(unique_Y):                
    #             indY = groups_idx_Y[j]                
    #             order_ty = order_delta[ty]                
    #             m_RS = delta_inv[order_tx, order_R] * delta_inv[order_S, order_ty]
    #             M[indX[:, None], indY] = m_RS
    #     return M

    @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def _matrix_dots(self, R, S):
        delta_inv = self.deltainv
        # print('dots')
        # print(delta_inv)
        X = Y = self.X_train
        task_col = self.task_info
        order_delta = self.order_delta
        nX = X.shape[0]
        nY = Y.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        # unique_X, groups_idx_X = npi.group_by(task_X,
        #                                           np.arange(nX))
        # unique_Y, groups_idx_Y = npi.group_by(task_Y,
        #                                       np.arange(nY))

        unique_X, groups_idx_X = self.unique, self.groups_idx
        unique_Y, groups_idx_Y = self.unique, self.groups_idx
        vX = np.zeros((nX, 1))    

        order_R = order_delta[R]
        order_S = order_delta[S]
        for i, tx in enumerate(unique_X):
            indX = groups_idx_X[i]
            order_tx = order_delta[tx]
            vX[indX] = delta_inv[order_tx, order_R]
            # order_R = order_delta[float(R)]


        vY = np.zeros((nY, 1)) 
        for j, ty in enumerate(unique_Y):                
            indY = groups_idx_Y[j]                
            order_ty = order_delta[ty]                
            vY[indY] = delta_inv[order_S, order_ty]

        M = (1 - self.lamb)**2 * (vX[self.svm.support_] @ vY[self.svm.support_].T)
        return M

    # # @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    # def _matrix_dots_full(self, R, S):
    #     delta_inv = self.deltainv
    #     # print('dots')
    #     # print(delta_inv)
    #     X = Y = self.X_train
    #     task_col = self.task_info
    #     order_delta = self.order_delta
    #     nX = X.shape[0]
    #     nY = Y.shape[0]
    #     task_X = X[:, task_col]
    #     task_Y = Y[:, task_col]
    #     # unique_X, groups_idx_X = npi.group_by(task_X,
    #     #                                           np.arange(nX))
    #     # unique_Y, groups_idx_Y = npi.group_by(task_Y,
    #     #                                       np.arange(nY))

    #     unique_X, groups_idx_X = self.unique, self.groups_idx
    #     unique_Y, groups_idx_Y = self.unique, self.groups_idx
    #     vX = np.zeros((nX, 1))    

    #     order_R = order_delta[R]
    #     order_S = order_delta[S]
    #     for i, tx in enumerate(unique_X):
    #         indX = groups_idx_X[i]
    #         order_tx = order_delta[tx]
    #         vX[indX] = delta_inv[order_tx, order_R]
    #         # order_R = order_delta[float(R)]


    #     vY = np.zeros((nY, 1)) 
    #     for j, ty in enumerate(unique_Y):                
    #         indY = groups_idx_Y[j]                
    #         order_ty = order_delta[ty]                
    #         vY[indY] = delta_inv[order_S, order_ty]

    #     M = vX @ vY.T
    #     return M

    @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def distances_between_tasks(self, G_train):
        reg = self.svm
        alpha = reg.dual_coef_.flatten()


        tasks = self.unique
        T = len(tasks)
        self.dots = np.zeros([T, T])
        self.dots_norm = np.zeros([T, T])
        self.dists = np.zeros([T, T])
        self.dists_norm = np.zeros([T, T]) 
        for i, tR in enumerate(tasks):
            for j, tS in enumerate(tasks):
                if j >= i:
                    M_RS = self._matrix_dots(tR, tS)
                    G_train_sup = G_train[reg.support_, reg.support_[:,None]]
                    dot = alpha.T @ np.multiply(M_RS, G_train_sup) @ alpha

                    # if S==R and dot < 0:
                    #     # print(R, S)
                    #     # print(dot)
                    #     # print('M_RS')
                    #     # print(M_RS)
                    #     # print('G_train')
                    #     # print(G_train)
                    #     # print('G_dist')
                    #     # print(G_dist)

                    order_S = self.order_delta[tS]
                    order_R = self.order_delta[tR]
                    self.dots[order_S, order_R] = self.dots[order_R, order_S] = dot
                    self.dots_norm[order_S, order_R] = self.dots_norm[order_R, order_S] = dot
        # print(self.dots)
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                self.dists[i, j] = self.dots[i, i] + self.dots[j, j] - 2 * self.dots[i, j]
                # print('dif:', self.dists[i, j])
                # print('sum:', (self.dots[i, i] + self.dots[j, j]))
                # print('quot:', self.dists[i, j]/(self.dots[i, i] + self.dots[j, j]))
                self.dists_norm[i, j] = self.dists[i, j]/(self.dots[i, i] + self.dots[j, j])
        return self.dists, self.dots
    
    # @timer
    def _optimize_laplacian_entropy(self, G_train, dists):
        # B computation
        num = np.exp(-(self.nu/self.mu) * dists) # 1e-16 * np.ones(dists.shape) # añadimos el 0 máquina
        # np.fill_diagonal(num, 0) # Se comenta para añadir la diagonal 
        den = np.sum(num, axis=1)
        B = num/den[:, None]

        # # B stable computation
        # den = np.zeros(dists.shape)
        # for t in range(dists.shape[0]):
        #     exponent = dists - dists[t, :]
        #     print('T: ', t)
        #     print(exponent)
        #     # np.fill_diagonal(exponent, 0)
        #     den += np.exp((1 / self.mu) * exponent)
        # # print('den')
        # # print(den)
        # B =  -1. / den
        # print('B despues')
        # print(B)

        # complete B
        L = deepcopy(-B)
        for i in range(len(self.order_delta)):
            L[i, i] = np.sum(B[i, :]) - B[i, i]
        self.B = B
        self.laplacian = L

    # def _optimize_laplacian_heuristic(self, G_train, dists):
    #     # print('dists')
    #     # print(dists)
    #     # dists_norm = (dists.T / np.sum(dists, axis=1)).T
    #     # print(dists_norm)

    #     T = dists.shape[0]

        
    #     B = 1 / dists

    #     # B = 1/(dists + np.mean(dists))

    #     # print(B)
    #     # for i, row in enumerate(dists):
    #     #     idx = (row < TOL)
    #     #     if len(idx) > 0:
    #     #         B[i, idx] = 1 / len(idx)
    #     #         B[i, ~idx] = 0
    #     #     else:
    #     #         B[i, :] = 1/dists[i, :]

    #     # print(B)
        
    #     for i, ti in enumerate(self.unique):
    #         B[i, i] = 0                
    #         B[i, :] /= np.sum(B[i, :])

    #     B = self.mu * np.identity(T) + (1 - self.mu) * B
        
    #     # complete L
    #     L = deepcopy(-B)
    #     indices = np.arange(len(self.order_delta))
    #     for i in range(len(self.order_delta)):
    #         L[i, i] = np.sum(B[i, :]) - B[i, i]
    #     self.B = B
    #     self.laplacian = L

    def _optimize_laplacian_heuristic(self, G_train, dists):
        # print('dists')
        # print(dists)
        # dists_norm = (dists.T / np.sum(dists, axis=1)).T
        # print(dists_norm)

        T = dists.shape[0]

        
        B = 1 / dists

        # B = 1/(dists + np.mean(dists))

        # print(B)
        # for i, row in enumerate(dists):
        #     idx = (row < TOL)
        #     if len(idx) > 0:
        #         B[i, idx] = 1 / len(idx)
        #         B[i, ~idx] = 0
        #     else:
        #         B[i, :] = 1/dists[i, :]

        # print(B)
        
        for i, ti in enumerate(self.unique):
            B[i, i] = 0
        B = B / np.sum(B)

        # complete L
        L = deepcopy(-B)
        indices = np.arange(len(self.order_delta))
        for i in range(len(self.order_delta)):
            L[i, i] = np.sum(B[i, :]) - B[i, i]
        self.B = B
        self.laplacian = L

    def _optimize_laplacian_cos(self, G_train, dists, dots):
        # print('dists')
        # print(dists)
        # dists_norm = (dists.T / np.sum(dists, axis=1)).T
        # print(dists_norm)

        T = dists.shape[0]

        
        B = np.zeros(dists.shape)
        
        for i, ti in enumerate(self.unique):
             for j, tj in enumerate(self.unique):
                B[i, j] = 1 - (dists[i, j] / (dots[i, i] + dots[j, j])) 

        # complete L
        L = deepcopy(-B)
        indices = np.arange(len(self.order_delta))
        for i in range(len(self.order_delta)):
            L[i, i] = np.sum(B[i, :]) - B[i, i]
        self.B = B
        self.laplacian = L
    
    def optimize_laplacian(self, G_train, dists=None, dots=None):
        if dists is None:
            dists, dots = self.distances_between_tasks(G_train)
        if self.opt == 'heuristic':
            # print('dists')
            # print(dists)
            B = -1/dists
            
            # print('dots')
            # print(self.dots)
            for i, ti in enumerate(self.unique):
                B[i, i] = 0                
                B[i, :] /= -np.sum(B[i, :])
                B[i, i] = -np.sum(B[i, :])
            # print(B)
            self.B_hist_.append(B)
            self.delta = B + B.T
            # self.delta /= np.sum(np.abs(self.delta)) # rho in [0, 1]
        elif self.opt == 'entropy':
            self._optimize_laplacian_entropy(G_train, dists)
        elif self.opt == 'cos':
            self._optimize_laplacian_cos(G_train, dists, dots)
        else:
            print('No such optimization method')
            exit()

        self.delta = self.laplacian + self.laplacian.T

        # Project delta eigenvalues into the positive cone (Carlos Alaiz)
        w, v = np.linalg.eig((self.delta + self.delta.T) / 2)
        pr_delta = np.dot(np.dot(v, np.maximum(np.diag(np.real(w)), 0)), v.T)
        self.delta = pr_delta

    def plot_history(self, start=None, stop=None, include='all', step='all', figsize=(10, 6), with_params=True, include_sum=False):
        iter = range(len(self.score_hist_))
        
        if start is None:
            start = 0
        if stop is None:
            stop = len(iter)

        if step == 'all':
            step = 1
        elif step == 'w':
            step = 2
            if start % 2 != 0:
                start = start + 1
        elif step == 'A':
            step = 2
            if start % 2 == 0:
                start = start + 1
        else:
            raise Exception('{} is not a valid value for step'.format(step))
        
        all_measures = ['score', 'reg', 'objf', 'ent']
        if include == 'all':
            include_ = all_measures
        else:
            if isinstance(include, list):
                for m in include:
                    if m not in all_measures:
                        raise Exception('{} is not a valid measure for include', m)
                include_ = include
            elif isinstance(include, str):
                if include not in all_measures:
                        raise Exception('{} is not a valid measure for include', include)
                include_ = [include]
            else:
                raise Exception('{} is not a valid type for include'.format(type(include)))

        
        plt.figure(figsize=figsize)

        sum_hist = np.zeros(len(iter))
        for m in include_:
            hist = np.array(getattr(self, '{}_hist_'.format(m)))
            if with_params is True:
                if m == 'score':
                    hist = self.C * hist
                elif m == 'ent':
                    hist = (self.mu / 2) * hist
                if m == 'ent':
                    sum_hist = sum_hist - hist
                else:
                    sum_hist = sum_hist + hist

            plt.plot(iter[start:stop:step], hist[start:stop:step] , marker='.', label=m)

        if include_sum:
                plt.plot(iter[start:stop:step], sum_hist[start:stop:step] , marker='.', label='sum')
        plt.xticks(iter[start:stop:step], iter[start:stop:step])

        ax = plt.gca()
        [t.set_color('brown') if int(t.get_text())%2==0 else t.set_color('darkgrey') for i, t in enumerate(ax.xaxis.get_ticklabels())]
        plt.legend()
        plt.tight_layout()
        # plt.title('Objective Function History')


    def show_matrix_hist_iter(self, iter, normalize=True):
        # Get the smaller and larger elements not in the diagonal

        distsnorm_hist_ent = [normalize_dists(dists, dots) for dists, dots in zip(self.dists_hist_, self.dots_hist_)]
        tasks = self.unique
        
        remove_diag_bool = True
        if normalize is True:
            hist = distsnorm_hist_ent
        else:
            hist = self.dists_hist_
        if remove_diag_bool:
            hist = [remove_diag(dots) for dots in hist]
        
        data = np.array(hist).flatten()
        min_data = data.min()#np.partition(data, T*max_iter_ext )[T*max_iter_ext]
        max_data = data.max()

        range_data = max_data - min_data
        per = 0.1
        bound_inf = min_data - per * range_data
        bound_sup = max_data + per * range_data
        
        fig, ax = plt.subplots(figsize=(12, 8))

        im, cbar = heatmap(hist[iter], tasks, tasks, ax=ax,
                        vmin=bound_inf, vmax=bound_sup,
                        cmap="bwr", cbarlabel=r'$|| w_r - w_s ||$')
        # texts = annotate_heatmap(im, valfmt="{x:.3f}")
        
        fig.tight_layout()

        # plt.title('Weights Distances - Iter {}'.format(iter), pad=10)

    
    def show_laplacian_hist_iter(self, iter):
        # Get the smaller and larger elements not in the diagonal

        tasks = self.unique
        
        remove_diag_bool = False
        hist = self.laplacian_hist_
        if remove_diag_bool:
            hist = [remove_diag(dots) for dots in hist]
        
        data = np.array(hist).flatten()
        min_data = data.min()#np.partition(data, T*max_iter_ext )[T*max_iter_ext]
        max_data = data.max()

        range_data = max_data - min_data
        per = 0.1
        bound_inf = min_data - per * range_data
        bound_sup = max_data + per * range_data
        
        fig, ax = plt.subplots(figsize=(12, 8))

        im, cbar = heatmap(hist[iter], tasks, tasks, ax=ax,
                        vmin=bound_inf, vmax=bound_sup,
                        cmap="bwr", cbarlabel=r'$L_{rs}$')
        # texts = annotate_heatmap(im, valfmt="{x:.3f}")
        
        fig.tight_layout()

        plt.title('Laplacian Matrix - Iter {}'.format(iter), pad=10)

    
    def show_B_hist_iter(self, iter, task_type='str'):
        # Get the smaller and larger elements not in the diagonal

        tasks = self.unique
        
        remove_diag_bool = False
        hist = self.B_hist_
        if remove_diag_bool:
            hist = [remove_diag(dots) for dots in hist]
        
        data = np.array(hist).flatten()
        min_data = data.min()#np.partition(data, T*max_iter_ext )[T*max_iter_ext]
        max_data = data.max()

        range_data = max_data - min_data
        per = 0.1
        bound_inf = 0 # min_data - per * range_data
        bound_sup = 1 # max_data + per * range_data
        
        fig, ax = plt.subplots(figsize=(12, 8))

        if task_type == 'str':
            task_labels = tasks.astype(str)
        elif task_type == 'int':
            task_labels = tasks.astype(int)
        else:
            raise Exception('{} is not a valid task_type'.format(task_type))

        im, cbar = heatmap(hist[iter], task_labels, task_labels, ax=ax,
                        vmin=bound_inf, vmax=bound_sup,
                        cmap="bwr", cbarlabel=r'$A_{rs}$')
        # texts = annotate_heatmap(im, valfmt="{x:.3f}")
        
        fig.tight_layout()

        # plt.title('Weight Matrix - Iter {}'.format(iter), pad=10)




class extInFusedMTLSVC(extInFusedMTLSVM):
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, task_info=None,
                 decision_function_shape='ovo', random_state=None, nu=1.0, max_iter_ext=-1,
                 opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0, inner_kernel=True):
        super(extInFusedMTLSVC, self).__init__(C, kernel, degree, gamma,
                                     coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, nu, task_info, max_iter_ext,
                                     opt, tol_ext, delta, order_delta, mu, 
                                     ind_reg, lamb, inner_kernel)
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.svm = SVC(self.C, kernel, self.degree, gamma, self.coef0,
                              self.shrinking, self.probability, self.tol, self.cache_size,
                              self.class_weight, self.verbose, self.max_iter,
                              self.decision_function_shape, self.random_state)
        return super().fit(X, y, task_info, **kwargs)
    
    def decision_function(self, X, G=None):
        if G is None:
            task_col = self.task_info
            X_train_data = np.delete(self.X_train, task_col, axis=1).astype(float)
            X_test_data = np.delete(X, task_col, axis=1).astype(float)

            G_test_standard = self._apply_kernel(self.kernel, X_test_data, X_train_data, self.gamma)
            G_test_fused = self._mtl_kernel_fused(X, self.X_train, self.kernel, self.gamma, self.delta, self.task_info)

            G_test = self.lamb**2 * G_test_standard + (1 - self.lamb)**2 * G_test_fused
        else:
            G_test = G
        return self.svm.decision_function(G_test)

    def score(self, X, y, G=None, sample_weight=None, scoring=None):
        pred = self.decision_function(X, G=G)
        return len(self.X_train) * hinge_loss(y, pred)


class extInFusedMTLSVR(extInFusedMTLSVM):
    """docstring for mtlSVM."""
    def __init__(self, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, nu=1.0, task_info=None, max_iter_ext=-1, opt='entropy',
                 tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0, inner_kernel=True):
        super(extInFusedMTLSVR, self).__init__(C, kernel, degree, gamma,
                                     coef0, shrinking, tol, cache_size,
                                     verbose, max_iter, nu, task_info, max_iter_ext,
                                     opt, tol_ext, delta, order_delta, mu, ind_reg, 
                                     lamb, inner_kernel)
        self.epsilon = epsilon


    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.svm = SVR(kernel, self.degree, gamma, self.coef0,
                       self.tol, self.C, self.epsilon, self.shrinking, self.cache_size,
                       self.verbose, self.max_iter)
        return super().fit(X, y, task_info, **kwargs)

    def score(self, X, y, G=None, sample_weight=None, scoring=None):
        pred = self.predict(X, G=G)
        return epsilon_insensitive_error(y, pred, self.epsilon)

    def loss_function(y_true, y_pred):
        return hinge_loss(y_true, y_pred)

    def decision_function(self, X, G=None):
        return self.predict(X, G=G)
    