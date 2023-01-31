from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH
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
# from notebooks.utils import get_train, timer, profile
import matplotlib.pyplot as plt
import matplotlib

from sklearn.base import clone

from icecream import ic
import time

import sys

from scipy.stats import entropy

from mtlskl.model.svm.adapglmtl.utils import *


aux_dir = 'aux_files'



class AdapGLKernelModel(BaseEstimator):
    'init'
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', cgamma=None, sgamma=None, coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, nu=1.0, task_info=None,
                 max_iter_ext=-1, opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.cgamma = cgamma
        self.sgamma = sgamma
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

    
    
    def compute_objf(self, X, y, G_train_standard_common, G_train_graphL, dists, dots, lap_update=False):
        # print('---------compute objf -------')

        # reg
        alpha = self.estim_.dual_coef_.T
        reg_common = self.lamb**2 * (alpha.T @ G_train_standard_common[self.estim_.support_, self.estim_.support_[:, None]] @ alpha)[0][0]
        # G_train_graphL = mtl_kernel_graphlap(X, X, self.kernel, self.gamma, self.deltainv, 
        #                                   self.order_delta, self.task_info, G=G_train_standard)
        # G_train = (1-self.lamb)**2 * G_train_graphL
        
        # reg = (1-self.lamb)**2 * (alpha.T @ G_train_graphL[self.estim_.support_, self.estim_.support_[:, None]] @ alpha)[0][0]         
        # # print('Reg1', reg)
        # reg = (self.nu * np.sum(self.A * dists) + self.nuaux * np.trace(dots)) # (1-self.lamb)**2  está en dots
        # # print('Reg2', reg)
        
        if not lap_update:
            reg = (1-self.lamb)**2 * (alpha.T @ G_train_graphL[self.estim_.support_, self.estim_.support_[:, None]] @ alpha)[0][0]       
            # print('Reg1', reg)
        else:
            reg = (self.nu * np.sum(self.A * dists) + self.nuaux * np.trace(dots)) # (1-self.lamb)**2  está en dots
            # print('Reg2', reg)

        

        # print('A')
        # print(self.A)

        reg = reg + reg_common

        self.reg_hist_.append(reg)
        
        # entropy
        ent = rows_entropy(self.A)
        self.ent_hist_.append(ent)
        # print('entropy:')
        # print(ent)
            
        # print('Prueba')
        # print(self.lamb)
        # print(-self.mu *ent + self.nu * np.sum(self.A * dists) )

        # Score
        G_train = self.lamb**2 * G_train_standard_common + (1-self.lamb)**2 * G_train_graphL
        # print(alpha.shape)
        # print(G_train[:, self.estim_.support_].shape)
        pred = G_train[:, self.estim_.support_] @ alpha + self.intercept_
        pred = pred.flatten()

        # err = epsilon_insensitive_error(y, pred, self.epsilon)
        # score = np.clip(err, 0, None).sum()
        # print('score 1', score)

        score = self.score(X, y, G=G_train)
        self.score_hist_.append(score)
        # print('score 2', score)

        objf = self.C * score + (1./2) * reg - (self.mu / 2) * ent

        # print('Score:', self.C * score)
        # print('Reg:',  reg )
        # print('Ent:', self.mu * ent)
        # print('Score + Reg:', self.C * score + reg)
        # print('Reg - Entropy:', reg - self.mu * ent)
        # print('Objf: ', objf)

        return objf

    def _get_GLkernelmatrix(self, X, Y, G, task_info):
        return mtl_kernel_graphlap(X, Y, self.kernel, self.gamma, self.deltainv,
                                   self.order_delta, task_info, G)


    def _mtl_kernel_train(self, G_train_standard_common, G_train_graphL):
        G_train = self.lamb**2 * G_train_standard_common + (1-self.lamb)**2 * G_train_graphL
        return G_train
    
    # @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def fit_no_gsearch(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        self.nuaux = 1

        # X = X[:1000, :]
        # y = y[:1000]
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
            self.gamma_ = 1
        else:
            self.gamma_ = self.gamma


        if self.cgamma is None:
            self.cgamma_ = self.gamma_
        else:
            self.cgamma_ = self.cgamma
        
        if self.sgamma is None:
            self.sgamma_ = self.gamma_
        else:
            self.sgamma_ = self.sgamma
        
        if 'delta' in kwargs:
            self.delta = kwargs['delta']
        else:
            T = len(self.unique)
            B = np.ones((T, T))
            self.A = B/T

            # B = np.random.random(size=(T, T))
            # self.A = B / np.sum(B, axis=1)

            self.A = (self.A + self.A.T)/2
            
            # B = np.identity(T) # Prueba con la identidad

            # L = deepcopy(-B)
            # for i in range(T):
            #     L[i, i] = np.sum(B[i, :]) - B[i, i]
            # # B = B / np.sum(np.abs(B))
            # self.A = B
            # self.laplacian = L
            # self.delta = self.laplacian + self.laplacian.T

        if 'order_delta' in kwargs:
            self.order_delta = kwargs['order_delta']
        else:
            self.order_delta = dict(zip(self.unique, range(len(self.unique))))

        self.laplacian = complete_laplacian(self.A, self.order_delta)
        self.delta = self.laplacian + self.laplacian.T
        self.deltainv = get_deltainv(self.delta, self.nu)

        if 'distfun' in kwargs:
            distances_comp = eval("distances_"+kwargs['distfun'])
        else:
            if 'cupy' in sys.modules:
                distances_comp = distances_loop_numpy # distances_divide_cupy 
            else:
                distances_comp = distances_loop_numpy

        den = int(len(self.unique) / 20) + 1 # empirically tried for eolo, maximum of 20 tasks
        

        self.deltainv_hist_ = []
        self.delta_hist_ = []
        self.laplacian_hist_ = []
        self.A_hist_ = []
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
        G_train_standard_common = apply_kernel(self.kernel, X_data, X_data, self.cgamma_)
        G_train_standard_specific = apply_kernel(self.kernel, X_data, X_data, self.sgamma_)

        # print(self.max_iter_ext)
        iter = 0
        stopCondition = False
        objf = 0    
        
        while not stopCondition:
            # print('Delta')
            # print(self.delta)
            # print('-----------------------------------ITER > {} -------------------------------------------'.format(iter))  
            # Delta Fixed and Optimize SVM
            
            G_train_graphL = self._get_GLkernelmatrix(X, X, G_train_standard_specific, task_info)
            # print(self.deltainv)
            
            G_train = self._mtl_kernel_train(G_train_standard_common, G_train_graphL)
                
            self.estim_ = clone(self.estim)
            self.estim_.fit(G_train, y, sample_weight)
            self.intercept_ = self.estim_.intercept_
            
            objf = self.compute_objf(X, y, G_train_standard_common, G_train_graphL, None, None)


            #Stopping Condition
            stopCondition = prev_objf - objf < self.tol_ext if iter > 0 else False
            stopCondition = stopCondition or ((iter >= self.max_iter_ext) if self.max_iter_ext > -1 else False)
            if not stopCondition:
                # print('------A-----')
                self.objf_hist_.append(objf)
                prev_objf = objf
                old_estim = self.estim_
                self.support_ = self.estim_.support_
                self.support_vectors_ = self.estim_.support_vectors_
                self.dual_coef_ = self.estim_.dual_coef_

                # print(self.dual_coef_)
                

                
                self.sample_weight = sample_weight
                self.task_info = task_info

                self.deltainv_hist_.append(self.deltainv)
                self.delta_hist_.append(self.delta)

                self.A_hist_.append(self.A)
                self.laplacian_hist_.append(self.laplacian)
                
                self.dual_coef_hist_.append(self.dual_coef_)
                self.support_hist_.append(self.support_)
                self.intercept_hist_.append(self.intercept_)
                # W fixed and optimize Delta
                self.delta_old = self.delta
                self.deltainv_old = self.deltainv
                # print('Compute Distances')
                # print(distances_comp)
                dists, dots = distances_comp(G=G_train_standard_specific,
                                             dual_coef=self.estim_.dual_coef_.flatten(),
                                             support_idx=self.estim_.support_,
                                             unique=self.unique,
                                             groups_idx=self.groups_idx,
                                             delta_inv=self.deltainv,
                                             order_delta=self.order_delta,
                                             lamb=self.lamb,
                                             den=den)


                # print('dists')
                # print(dists)
                
                # print('Optimize Laplacian')
                self.A = self.optimize_adjacency(G_train_standard_specific, dists, dots)

                self.laplacian = complete_laplacian(self.A, self.order_delta)

                self.delta = self.laplacian + self.laplacian.T
                # # Project delta eigenvalues into the positive cone (Carlos Alaiz)
                # w, v = np.linalg.eig(self.delta)
                # self.delta = np.dot(np.dot(v, np.maximum(np.diag(np.real(w)), 0)), v.T)
                self.deltainv = get_deltainv(self.delta, self.nu)

                self.dots_hist_.append(dots)
                self.dists_hist_.append(dists)

                objf = self.compute_objf(X, y, G_train_standard_common, G_train_graphL, dists, dots, lap_update=True)
                # print('------B-----')
                self.objf_hist_.append(objf)
                
                
            else:
                if self.max_iter_ext > 0:
                    self.objf_hist_ = self.objf_hist_[:-1]
                    self.score_hist_ = self.score_hist_[:-1]
                    self.reg_hist_ = self.reg_hist_[:-1]
                    self.ent_hist_ = self.ent_hist_[:-1]
                    # self.A_hist_ = self.A_hist_[:-1]
                    self.estim_ = old_estim
                    self.delta = self.delta_old
                    self.deltainv = self.deltainv_old
                break
            iter+=1
        # if iter > 0:
        #     objf = self.compute_objf(X, y, G_train_standard_common, G_train_graphL, dists, dots)
        self.total_iter = iter
        return self
    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        return self.fit_no_gsearch(X, y, task_info, sample_weight, **kwargs)


    def predict(self, X, G=None):
        if G is None:
            task_col = self.task_info
            X_train_data = np.delete(self.X_train, task_col, axis=1).astype(float)
            X_test_data = np.delete(X, task_col, axis=1).astype(float)

            G_test_standard = apply_kernel(self.kernel, X_test_data, X_train_data, self.cgamma_)
            G_test_fused = mtl_kernel_graphlap(X, self.X_train, self.kernel,
                                            gamma=self.sgamma_,
                                            deltainv=self.deltainv,
                                            order_delta=self.order_delta,
                                            task_info=self.task_info)

            G_test = self.lamb**2 * G_test_standard + (1 - self.lamb)**2 * G_test_fused
        else:
            G_test = G

        # ic(G_test)
        pred = self.estim_.predict(G_test)

        return pred

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
    
    def optimize_adjacency(self, G_train, dists=None, dots=None):
        
        if self.opt == 'heuristic':
            self.A = optimize_adjacency_heur(dists, dots, self.order_delta, self.nu, self.mu)
        elif self.opt == 'entropy':
            self.A= optimize_adjacency_entropy(dists, dots, self.order_delta, self.nu, self.mu)
        elif self.opt == 'cos':
            self.A= optimize_adjacency_cos(dists, dots, self.order_delta, self.nu, self.mu)
        else:
            # print('No such optimization method')
            exit()

        self.A = (self.A + self.A.T)/2
        return self.A

        

    def plot_history(self, start=None, stop=None, include='all', step='all', figsize=(10, 6), with_params=True, include_sum=False):
        iter = range(len(self.objf_hist_))
        
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
        hist = self.A_hist_
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




