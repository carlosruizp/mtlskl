from os import EX_CANTCREAT
from sklearn.svm import SVC
from sklearn.svm import SVR
import sklearn.metrics.pairwise as pairwise
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import hinge_loss
from sklearn.base import BaseEstimator, clone
import numpy as np
import numpy_indexed as npi
import types
import pickle
from notebooks.utils import get_train, timer, profile
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy

from scipy.stats import entropy


aux_dir = 'aux_files'


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


class GlobalMTLSVM(BaseEstimator):
    'init'
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, nu=1.0, task_info=None,
                 max_iter_ext=-1, opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0, alpha=0.5):
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
        self.alpha = alpha
        

    
    
    def compute_objf(self, X, y, G_train):
        # error
        err_tot = np.sum(self.err_matrix * self.delta)

        # regularization
        reg_tot = 0
        reg_dic = {}
        for t in self.unique:
            alpha = self.svm_dic[t]._dual_coef_.T
            reg_ =  (alpha.T @ G_train[self.svm_dic[t].support_, self.svm_dic[t].support_[:, None]] @ alpha)[0][0]
            reg_dic[t] = reg_
            reg_tot += reg_

        # entropy
        # entropy
        ent = rows_entropy(self.delta)
        

        objf = self.C * err_tot + reg_tot + self.nu * ent
        self.reg_hist_.append(reg_tot)
        self.score_hist_.append(err_tot)
        self.ent_hist_.append(ent)
        self.objf_hist_.append(objf)


        return objf
    
    # @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
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
            # B = B / np.sum(np.abs(B))
            self.delta = self.alpha * np.identity(len(self.unique)) + (1 - self.alpha) * B

        if 'order_delta' in kwargs:
            self.order_delta = kwargs['order_delta']
        else:
            self.order_delta = dict(zip(self.unique, range(len(self.unique))))

        self.delta_hist_ = []
        self.err_matrix_hist_ = []

        self.score_hist_ = []
        self.reg_hist_ = []
        self.ent_hist_ = []
        self.objf_hist_ = []
        score = 0
        # print('FIT')
        # Get G_train
        G_train = self._apply_kernel(self.kernel, X_data, X_data, self.gamma)
        # self.y = y

        # print(self.max_iter_ext)
        iter = 0
        stopCondition = False
        objf = 0

        self.svm_dic = {}
        for t in self.unique:
            self.svm_dic[t] = clone(self.svm)
        
        self.support_ = {}
        self.support_vectors_ = {}
        self.dual_coef_ = {}
        self.intercept_ = {}
        self.sample_weight = {}
        self.err_matrix = np.zeros(self.delta.shape)

        # print(stopCondition)
        
        while not stopCondition:            
            # print('-----------------------------------ITER > {} -------------------------------------------'.format(iter))
            # print('Delta')
            # print(self.delta)
            for t in self.unique:
                # print(' task {}'.format(t))
                # define weights
                # print(" sample weight")
                sample_weight = np.zeros(X.shape[0])
                for u, u_idx in zip(self.unique, self.groups_idx):
                    u_order = self.order_delta[u]
                    t_order = self.order_delta[t]
                    sample_weight[u_idx] = self.delta[t_order, u_order] + np.finfo(float).tiny
                # print(sample_weight)

                # Delta Fixed and Optimize SVM
                self.svm_dic[t].fit(G_train, y, sample_weight)
                self.support_[t] = self.svm_dic[t].support_
                self.support_vectors_[t] = self.svm_dic[t].support_vectors_
                self.dual_coef_[t] = self.svm_dic[t].dual_coef_                
                self.intercept_[t] = self.svm_dic[t].intercept_
                self.sample_weight[t] = sample_weight


                # Compute errors
                pred = self.svm_dic[t].predict(G_train)

                t_order = self.order_delta[t]
                t_idx = self.groups_idx[t_order]
                for u in self.unique:
                    u_order = self.order_delta[u]
                    u_idx = self.groups_idx[u_order]
                    loss_tu = self.loss_function(y[u_idx].flatten(), pred[u_idx])
                    # print(loss_tu)
                    self.err_matrix[t_order, u_order] = loss_tu

                # print(self.err_matrix)

            # hist
            self.delta_hist_.append(self.delta)
            self.err_matrix_hist_.append(self.err_matrix)

            # print(self.svm_dic)
            prev_objf = objf
            # print('Fitted SVM')
            
            objf = self.compute_objf(X, y, G_train)
            # print(objf)

            # print('Update delta')            

            #Stopping Condition
            stopCondition = np.abs(prev_objf - objf) < self.tol_ext
            stopCondition = stopCondition or ((iter >= self.max_iter_ext) if self.max_iter_ext > -1 else False)
            # print(stopCondition)
            if not stopCondition:
                # W fixed and optimize Delta
                # self.delta_old = self.delta
                self.optimize_delta(G_train)
                objf = self.compute_objf(X, y, G_train)
            else:
                break
            iter+=1
        self.total_iter = iter
            # print((iter >= self.max_iter_ext) if self.max_iter_ext > 0 else False)
    
    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        return self.fit_no_gsearch(X, y, task_info, sample_weight, **kwargs)


    def predict(self, X, G=None):
        task_col = self.task_info
        unique, groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(X.shape[0]))
        X_data = np.delete(X, task_col, axis=1).astype(float)
        if G is None:
            task_col = self.task_info
            X_train_data = np.delete(self.X_train, task_col, axis=1).astype(float)
            X_test_data = np.delete(X, task_col, axis=1).astype(float)

            G_test_standard = self._apply_kernel(self.kernel, X_test_data, X_train_data, self.gamma)

            G_test =  G_test_standard
        else:
            G_test = G

        pred = np.zeros(X.shape[0])
        for t in self.unique:
            t_order = self.order_delta[t]
            test_idx = groups_idx[t_order]
            pred_t = self.svm_dic[t].predict(G_test[test_idx, :])
            pred[test_idx] = pred_t
        
        return pred

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

    def _optimize_delta_entropy(self):
        # print('Err matrix')
        # print(self.err_matrix)
        # print(-(self.C * (1 - self.alpha)/self.mu) * self.err_matrix)
        # B computation
        num = np.exp(-(self.C * (1 - self.alpha)/self.mu) * self.err_matrix)
        # np.fill_diagonal(num, 0) # Se comenta para añadir la diagonal 
        # Ponemos esta correccion para cuando los números del exponente son muy pequeños (muy negativos) y toda la fila es 0
        for rownum, rowerr in zip(num, self.err_matrix):
            if (rownum == 0).all():
                # print(np.argmin(rowerr))
                rownum[np.argmin(rowerr)] = 1
        den = np.sum(num, axis=1)
        B = num/den[:, None]
        return B

    
    def optimize_delta(self, G_train):
        if self.opt == 'heuristic':
            pass
        elif self.opt == 'entropy':
            B = self._optimize_delta_entropy()
        else:
            raise Exception('No such optimization method')
        self.delta = self.alpha * np.identity(len(self.unique)) + (1 - self.alpha) * B

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


    def show_err_matrix_hist_iter(self, iter, task_type='str'):
        # Get the smaller and larger elements not in the diagonal

        tasks = self.unique
        
        remove_diag_bool = False
        hist = self.err_matrix_hist_
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

        if task_type == 'str':
            task_labels = tasks.astype(str)
        elif task_type == 'int':
            task_labels = tasks.astype(int)
        else:
            raise Exception('{} is not a valid task_type'.format(task_type))

        im, cbar = heatmap(hist[iter], task_labels, task_labels, ax=ax,
                        vmin=bound_inf, vmax=bound_sup,
                        cmap="YlGn", cbarlabel=r'$A_{rs}$')
        texts = annotate_heatmap(im, valfmt="{x:.3f}")
        
        fig.tight_layout()

        # plt.title('Weight Matrix - Iter {}'.format(iter), pad=10)

    
    def show_delta_hist_iter(self, iter, task_type='str'):
        # Get the smaller and larger elements not in the diagonal

        tasks = self.unique
        
        remove_diag_bool = False
        hist = self.delta_hist_
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
                        cmap="YlGn", cbarlabel=r'$A_{rs}$')
        texts = annotate_heatmap(im, valfmt="{x:.3f}")
        
        fig.tight_layout()

        # plt.title('Weight Matrix - Iter {}'.format(iter), pad=10)




class GlobalMTLSVC(GlobalMTLSVM):
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, task_info=None,
                 decision_function_shape='ovo', random_state=None, nu=1.0, max_iter_ext=-1,
                 opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0, alpha=0.5):
        super(GlobalMTLSVC, self).__init__(C, kernel, degree, gamma,
                                     coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, nu, task_info, max_iter_ext,
                                     opt, tol_ext, delta, order_delta, mu, ind_reg, lamb,
                                     alpha)
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


class GlobalMTLSVR(GlobalMTLSVM):
    """docstring for mtlSVM."""
    def __init__(self, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, nu=1.0, task_info=None, max_iter_ext=-1, opt='entropy',
                 tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0, alpha=0.5):
        super(GlobalMTLSVR, self).__init__(C, kernel, degree, gamma,
                                     coef0, shrinking, tol, cache_size,
                                     verbose, max_iter, nu, task_info, max_iter_ext,
                                     opt, tol_ext, delta, order_delta, mu, ind_reg, lamb,
                                     alpha)
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

    def loss_function(self, y_true, y_pred):
        return epsilon_insensitive_error(y_true, y_pred, self.epsilon)

    def decision_function(self, X, G=None):
        return self.predict(X, G=G)
    


# class HeteroSVR(SVR):
#     """This class implements an SVR model with possible heterogeneous parameters"""
#     def __init__(self, kernel='rbf', degree=3, gamma='auto',
#                  coef0=0.0, tol=0.001, C=1.0,
#                  epsilon=0.1, shrinking=True, cache_size=200,
#                  verbose=False, max_iter=-1):
#         self.C = C
#         self.epsilon = epsilon
#         self.kernel = kernel
#         self.degree = degree
#         self.gamma = gamma
#         self.coef0 = coef0
#         self.shrinking = shrinking
#         self.tol = tol
#         self.cache_size = cache_size
#         self.verbose = verbose
#         self.max_iter = max_iter
    

#     def fit(self, X, y, sample_weight=None, **kwargs):
#         pass

#     def predict(X):
#         pass

#     def score(X, y):
#         pass
