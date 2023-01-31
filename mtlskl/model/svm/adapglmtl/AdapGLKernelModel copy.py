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

from icecream import ic

from scipy.stats import entropy

from kernel.adapGraphLap.utils import *


aux_dir = 'aux_files'



class AdapGLKernelModel(BaseEstimator):
    'init'
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, nu=1.0, task_info=None,
                 max_iter_ext=-1, opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0):
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

    
    
    def compute_objf(self, X, y, G_train_standard, G_train_graphL, dists, dots):
        print('---------compute objf -------')

        # reg
        alpha = self.estim.dual_coef_.T
        reg_common = self.lamb**2 * (alpha.T @ G_train_standard[self.estim.support_, self.estim.support_[:, None]] @ alpha)[0][0]
        # G_train_graphL = mtl_kernel_graphlap(X, X, self.kernel, self.gamma, self.deltainv, 
        #                                   self.order_delta, self.task_info, G=G_train_standard)
        G_train = (1-self.lamb)**2 * G_train_graphL
        
        reg = (alpha.T @ G_train[self.estim.support_, self.estim.support_[:, None]] @ alpha)[0][0]         
        print('Reg1', reg)

        G_train_graphL = mtl_kernel_graphlap(X, X, self.kernel, self.gamma, self.deltainv, 
                                          self.order_delta, self.task_info, G=G_train_standard)
        G_train = (1-self.lamb)**2 * G_train_graphL
        
        reg = (alpha.T @ G_train[self.estim.support_, self.estim.support_[:, None]] @ alpha)[0][0]         
        print('Reg2', reg)

        reg = (self.nu * np.sum(self.B * dists) + self.nuaux * np.trace(dots)) # (1-self.lamb)**2  está en dots
        print('Reg3', reg)

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
        G_train = self.lamb**2 * G_train_standard + (1-self.lamb)**2 * G_train_graphL
        # print(alpha.shape)
        # print(G_train[:, self.estim.support_].shape)
        # pred = G_train[:, self.estim.support_] @ alpha + self.intercept_
        # pred = pred.flatten()

        score = self.score(X, y, G=G_train)
        self.score_hist_.append(score)

        objf = self.C * score + (1./2) * reg - (self.mu / 2) * ent

        print('Score:', self.C * score)
        print('Reg:',  reg )
        print('Ent:', self.mu * ent)
        print('Score + Reg:', self.C * score + reg)
        print('Reg - Entropy:', reg - self.mu * ent)
        print('Objf: ', objf)
        
        self.objf_hist_.append(objf)

        return objf

    def _get_GLkernelmatrix(self, X, Y, G, task_info):
        return mtl_kernel_graphlap(X, Y, self.kernel, self.gamma, self.deltainv,
                                   self.order_delta, task_info, G)
    
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
            self.B = B/T
            # B = np.identity(T) # Prueba con la identidad

            # L = deepcopy(-B)
            # for i in range(T):
            #     L[i, i] = np.sum(B[i, :]) - B[i, i]
            # # B = B / np.sum(np.abs(B))
            # self.B = B
            # self.laplacian = L
            # self.delta = self.laplacian + self.laplacian.T
            

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
        # Get G_train
        G_train_standard = apply_kernel(self.kernel, X_data, X_data, self.gamma)

        # print(self.max_iter_ext)
        iter = 0
        stopCondition = False
        objf = 0

        self.laplacian = complete_laplacian(self.B, self.order_delta)
        self.delta = self.laplacian + self.laplacian.T
        self.deltainv = get_deltainv(self.delta, self.nu)

        
        while not stopCondition:
            # print('Delta')
            # print(self.delta)
            print('-----------------------------------ITER > {} -------------------------------------------'.format(iter))  
            # Delta Fixed and Optimize SVM
            
            G_train_graphL = self._get_GLkernelmatrix(X, X, G_train_standard, task_info)
            
            # print(self.deltainv)
            self.y = y
            
            G_train = self.lamb**2 * G_train_standard + (1-self.lamb)**2 * G_train_graphL
                
            self.estim.fit(G_train, self.y, sample_weight)
            self.support_ = self.estim.support_
            self.support_vectors_ = self.estim.support_vectors_
            self.dual_coef_ = self.estim.dual_coef_
            
            self.intercept_ = self.estim.intercept_
            self.sample_weight = sample_weight
            self.task_info = task_info

            self.deltainv_hist_.append(self.deltainv)
            self.delta_hist_.append(self.delta)

            self.B_hist_.append(self.B)
            self.laplacian_hist_.append(self.laplacian)
            
            self.dual_coef_hist_.append(self.dual_coef_)
            self.support_hist_.append(self.support_)
            self.intercept_hist_.append(self.intercept_)

            # prev_objf = objf
            # if ((iter < self.max_iter_ext) if self.max_iter_ext > -1 else True):
            #     print('Fitted SVM')
            #     dists, dots = distances_between_tasks(G=G_train_standard,
            #                                           dual_coef=self.estim.dual_coef_.flatten(),
            #                                           support_idx=self.estim.support_,
            #                                           unique=self.unique,
            #                                           groups_idx=self.groups_idx,
            #                                           delta_inv=self.deltainv,
            #                                           order_delta=self.order_delta,
            #                                           lamb=self.lamb)
            #     self.delta_old = self.delta
            #     self.deltainv_old = self.deltainv 
            #     # objf = self.compute_objf(X, y, G_train_standard, dists, dots)

            print('Update L')
            

            #Stopping Condition
            stopCondition = np.abs(prev_objf - objf) < self.tol_ext if iter > 0 else False
            stopCondition = stopCondition or ((iter >= self.max_iter_ext) if self.max_iter_ext > -1 else False)
            if not stopCondition:
                prev_objf = objf
                # W fixed and optimize Delta
                
                dists, dots = distances_between_tasks(G=G_train_standard,
                                                      dual_coef=self.estim.dual_coef_.flatten(),
                                                      support_idx=self.estim.support_,
                                                      unique=self.unique,
                                                      groups_idx=self.groups_idx,
                                                      delta_inv=self.deltainv,
                                                      order_delta=self.order_delta,
                                                      lamb=self.lamb)
                objf = self.compute_objf(X, y, G_train_standard, G_train_graphL, dists, dots)

                self.optimize_laplacian(G_train_standard, dists, dots) # changes self.delta
                # self.deltainv = get_deltainv(self.delta, self.nu)

                self.dots_hist_.append(dots)
                self.dists_hist_.append(dists)

                objf = self.compute_objf(X, y, G_train_standard, G_train_graphL, dists, dots)
                
            else:
                break
            iter+=1
        self.total_iter = iter
        return self
            # print((iter >= self.max_iter_ext) if self.max_iter_ext > 0 else False)
    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        return self.fit_no_gsearch(X, y, task_info, sample_weight, **kwargs)


    def predict(self, X, G=None):
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


        return self.estim.predict(G_test)

    def score(self, X, y, sample_weight=None, scoring=None):
        
        y_pred = self.predict(X)

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
    
    def optimize_laplacian(self, G_train, dists=None, dots=None):
        if dists is None:
            dists, dots = distances_between_tasks(G=G_train,
                                                  dual_coef=self.estim.dual_coef_.flatten(),
                                                  support_idx=self.estim.support_,
                                                  unique=self.unique,
                                                  groups_idx=self.groups_idx,
                                                  delta_inv=self.deltainv,
                                                  order_delta=self.order_delta,
                                                  lamb=self.lamb)
        if self.opt == 'heuristic':
            self.B= optimize_laplacian_heur(dists, dots, self.order_delta, self.nu, self.mu)
        elif self.opt == 'entropy':
            self.B= optimize_laplacian_entropy(dists, dots, self.order_delta, self.nu, self.mu)
        elif self.opt == 'cos':
            self.B= optimize_laplacian_cos(dists, dots, self.order_delta, self.nu, self.mu)
        else:
            # print('No such optimization method')
            exit()

        self.delta_old = self.delta
        self.deltainv_old = self.deltainv
        self.laplacian = complete_laplacian(self.B, self.order_delta)
        self.delta = self.laplacian + self.laplacian.T
        self.deltainv = get_deltainv(self.delta, self.nu)
        # Project delta eigenvalues into the positive cone (Carlos Alaiz)
        # w, v = np.linalg.eig(self.delta)
        # pr_delta = np.dot(np.dot(v, np.maximum(np.diag(np.real(w)), 0)), v.T)
        # self.delta = pr_delta


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
                        vmin=bound_inf, vmax=bound_sup, # cmap="bwr",
                        cbarlabel=r'$A_{rs}$')
        texts = annotate_heatmap(im, valfmt="{x:.3f}")
        
        fig.tight_layout()

        # plt.title('Weight Matrix - Iter {}'.format(iter), pad=10)




