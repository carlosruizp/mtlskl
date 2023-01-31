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

from scipy.stats import entropy

from kernel.adapGraphLap.utils import *

from icecream import ic


aux_dir = 'aux_files'



class extFusedMTLSVM(BaseEstimator):
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

    
    
    def compute_objf(self, X, y, G_train_standard, dists, dots):
        # print('MATRICES')
        # print(self.B)
        # print(self.laplacian)
        # print(self.delta)
        # print(self.delta_old)

        # reg
        # print('delta')
        # print(self.delta)
        alpha = self.svm._dual_coef_.T
        reg_common = self.lamb**2 * (alpha.T @ G_train_standard[self.svm.support_, self.svm.support_[:, None]] @ alpha)[0][0]
        G_train_graphL = self._mtl_kernel_fused(X, X, self.kernel, self.gamma, self.delta, 
                                            self.task_info, G_train_standard)
        G_train = (1-self.lamb)**2 * G_train_graphL
        
        # reg = (alpha.T @ G_train[self.svm.support_, self.svm.support_[:, None]] @ alpha)[0][0]         
        # print('Reg')
        # print(reg)

        reg = (self.nu * np.sum(self.B * dists) + self.nuaux * np.trace(dots)) # (1-self.lamb)**2  está en dots
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
        # print('delta old')
        # print(self.delta_old)
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
            # print('-----------------------------------ITER > {} -------------------------------------------'.format(iter))  
            # Delta Fixed and Optimize SVM       
            G_train_graphL = self._mtl_kernel_fused(X, X, self.kernel, self.gamma, self.delta,
                                             task_info, G_train_standard)

            
            # print(self.deltainv)
            self.y = y
            
            G_train = self.lamb**2 * G_train_standard + (1-self.lamb)**2 * G_train_graphL
            
            # print('FIT Model #################')
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
            if ((iter < self.max_iter_ext) if self.max_iter_ext > -1 else True):
                alpha = self.svm._dual_coef_.T
                dists, dots = self.distances_between_tasks(G_train_standard)
                self.delta_old = self.delta
                objf = self.compute_objf(X, y, G_train_standard, dists, dots)

            # print('Update L')
            

            #Stopping Condition
            stopCondition = np.abs(prev_objf - objf) < self.tol_ext
            stopCondition = stopCondition or ((iter >= self.max_iter_ext) if self.max_iter_ext > -1 else False)
            if not stopCondition:
                # W fixed and optimize Delta
                # self.delta_old = self.delta
                # print('Compute Distances')
                # alpha = self.svm._dual_coef_.T
                # dists, dots = self.distances_between_tasks(G_train_standard)
                # self.delta_old = self.delta
                # objf = self.compute_objf(X, y, G_train_standard, dists, dots)
                self.delta = self.optimize_laplacian(G_train_standard, dists, dots)
                self.dots_hist_.append(self.dots)
                self.dists_hist_.append(self.dists)
                # print('Optimize Laplacian ################')
                objf = self.compute_objf(X, y, G_train_standard, dists, dots)
            else:
                break
            iter+=1
        self.total_iter = iter
            # print((iter >= self.max_iter_ext) if self.max_iter_ext > 0 else False)
    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        # print('FIT')
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

    # @timer
    def _mtl_kernel_fused(self, X, Y, kernel, gamma, delta, task_info=-1, G_train=None):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info

        if self.ind_reg:
            delta_full = self.nu * delta + self.nuaux * np.identity(delta.shape[0])
        else:
            delta_full = delta

            
        # print(is_pos_semidef(delta))
        self.deltainv = np.linalg.inv(delta_full)


        order_delta = self.order_delta
        

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
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                order_tx = order_delta[tx]
                order_ty = order_delta[ty]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy
        if G_train is None:
            Q = self._apply_kernel(kernel, X_data, Y_data, gamma)
        else:
            Q = G_train

        Q_graphL = np.multiply(A, Q)
        
        return Q_graphL

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
        #     # print('T: ', t)
        #     # print(exponent)
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
            # print('No such optimization method')
            exit()

        delta = self.laplacian + self.laplacian.T

        # Project delta eigenvalues into the positive cone (Carlos Alaiz)
        w, v = np.linalg.eig((delta + delta.T) / 2)
        pr_delta = np.dot(np.dot(v, np.maximum(np.diag(np.real(w)), 0)), v.T)
        delta = pr_delta
        return delta

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




class extFusedMTLSVC(extFusedMTLSVM):
    """docstring for mtlSVM."""
    def __init__(self, C=1.0, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, task_info=None,
                 decision_function_shape='ovo', random_state=None, nu=1.0, max_iter_ext=-1,
                 opt='entropy', tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0):
        super(extFusedMTLSVC, self).__init__(C, kernel, degree, gamma,
                                     coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, nu, task_info, max_iter_ext,
                                     opt, tol_ext, delta, order_delta, mu, ind_reg, lamb)
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    
    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.svm = SVC(C=self.C, kernel=kernel, degree=self.degree, gamma=gamma, coef0=self.coef0,
                              shrinking=self.shrinking, probability=self.probability, tol=self.tol, cache_size=self.cache_size,
                              class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter,
                              decision_function_shape=self.decision_function_shape, random_state=self.random_state)
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


class extFusedMTLSVR(extFusedMTLSVM):
    """docstring for mtlSVM."""
    def __init__(self, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, nu=1.0, task_info=None, max_iter_ext=-1, opt='entropy',
                 tol_ext=1e-3, delta=None, order_delta=None, mu=1.0,
                 ind_reg=True, lamb=0):
        super(extFusedMTLSVR, self).__init__(C, kernel, degree, gamma,
                                     coef0, shrinking, tol, cache_size,
                                     verbose, max_iter, nu, task_info, max_iter_ext,
                                     opt, tol_ext, delta, order_delta, mu, ind_reg, lamb)
        self.epsilon = epsilon


    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.svm = SVR(kernel=kernel, degree=self.degree, gamma=gamma, coef0=self.coef0,
                       tol=self.tol, C=self.C, epsilon=self.epsilon, shrinking=self.shrinking, cache_size=self.cache_size,
                       verbose=self.verbose, max_iter=self.max_iter)
        return super().fit(X, y, task_info, **kwargs)

    def score(self, X, y, G=None, sample_weight=None, scoring=None):
        pred = self.predict(X, G=G)
        return epsilon_insensitive_error(y, pred, self.epsilon)

    def loss_function(y_true, y_pred):
        return hinge_loss(y_true, y_pred)

    def decision_function(self, X, G=None):
        return self.predict(X, G=G)
    