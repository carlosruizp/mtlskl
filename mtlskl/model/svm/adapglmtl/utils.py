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
from copy import deepcopy

from scipy.stats import entropy

# import cupy as cp
# from cupy.cuda.memory import OutOfMemoryError

from icecream import ic


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
    return np.clip(err, 0, None) #np.sum()

def hinge_error(ytrue, ypred):

    ytrue_flat = ytrue.flatten()
    ypred_flat = ypred.flatten()
    if ytrue_flat.shape != ypred_flat.shape:
        raise Exception('True value and prediction have not equal lengths')


    # ic(ytrue_flat * ypred_flat)
    # ic(1 - ytrue_flat * ypred_flat)

    err = np.clip(1 - ytrue_flat * ypred_flat, 0, None)   
    return err


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


def get_vsum(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb):
    
    T = len(unique)
    n_sv = len(support_idx)

    v_mat = np.zeros((T, T, n_sv))
    
    for i, tx in enumerate(unique):        
        indX = groups_idx[i]
        order_tx = order_delta[tx]
        for j, ty in enumerate(unique):
            indY = groups_idx[j]                
            order_ty = order_delta[ty]
            vX = np.zeros((n, 1))
            vX[indX] = delta_inv[order_tx, order_ty]
            v_mat[order_tx, order_ty] = vX[support_idx].flatten()
    vsum = v_mat.sum(axis=0)

    return vsum


# def dots_loop_numpy(R, S, n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb):
#     nY = nX = n
#     vX = np.zeros((nX, 1))
#     vY = np.zeros((nY, 1))
#     unique_X = unique
#     unique_Y = unique
#     groups_idx_X = groups_idx
#     groups_idx_Y = groups_idx
    

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

#     M = (1 - lamb)**2 * (vX[support_idx] @ vY[support_idx].T)
#     return M


def distances_loop_numpy(G, dual_coef, support_idx, unique, groups_idx, delta_inv, order_delta, lamb, **kwargs):

    T = len(unique)
    n = G.shape[0]
    dots = np.zeros([T, T])
    dists = np.zeros([T, T])
    alpha = dual_coef[:, None]
    G_sup = G[support_idx, support_idx[:,None]]
    
    vsum = get_vsum(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb)

    for i, tR in enumerate(unique):
        # ic(tR)
        order_R = order_delta[tR]
        vX = vsum[order_R][:, None]
        for j, tS in enumerate(unique):    
            # ic(tS)        
            if j >= i:
                # ic(i, j)
                order_S = order_delta[tS]                
                vY = vsum[order_S][:, None]
                M_RS = (1 - lamb)**2 * (vX @ vY.T)
                # ic(M_RS)
                dot = alpha.T @ np.multiply(M_RS, G_sup) @ alpha

                
                dots[order_S, order_R] = dots[order_R, order_S] = dot
    
    # DISTS
    dots_diag = np.diag(dots)[:, None]
    dists = dots_diag + dots_diag.T - 2*dots

    return dists, dots



def distances_loop_cupy(G, dual_coef, support_idx, unique, groups_idx, delta_inv, order_delta, lamb, **kwargs):

    T = len(unique)
    n = G.shape[0]
    dots = np.zeros([T, T])
    dists = np.zeros([T, T])
    alpha = dual_coef[:, None]

    G_sup = G[support_idx, support_idx[:,None]]

    vsum = get_vsum(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb)

    for i, tR in enumerate(unique):
        order_R = order_delta[tR]
        vX = vsum[order_R][:, None]
        for j, tS in enumerate(unique):            
            if j >= i:
                order_S = order_delta[tS]                
                vY = vsum[order_S][:, None]
                M_RS = (1 - lamb)**2 * cp.dot(vX, vY.T)                
                dot = cp.dot(cp.dot(alpha.T , np.multiply(M_RS, G_sup)), alpha)
                dots[order_S, order_R] = dots[order_R, order_S] = dot
   
   # DISTS
    dots_diag = np.diag(dots)[:, None]
    dists = dots_diag + dots_diag.T - 2*dots
    return cp.asnumpy(dists), cp.asnumpy(dots)


def dots_numpy(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb):
    
    T = len(unique)
    n_sv = len(support_idx)

    v_mat = np.zeros((T, T, n_sv))
    
    for i, tx in enumerate(unique):        
        indX = groups_idx[i]
        order_tx = order_delta[tx]
        for j, ty in enumerate(unique):
            indY = groups_idx[j]                
            order_ty = order_delta[ty]
            vX = np.zeros((n, 1))
            vX[indX] = delta_inv[order_tx, order_ty]
            v_mat[order_tx, order_ty] = vX[support_idx].flatten()
    v_sum = v_mat.sum(axis=0)

    dots_tensor = (1 - lamb)**2 * np.einsum('hi,jk->hjik', v_sum, v_sum)

    return dots_tensor

def distances_numpy(G, dual_coef, support_idx, unique, groups_idx, delta_inv, order_delta, lamb, **kwargs):

    T = len(unique)
    n = G.shape[0]
    alpha = dual_coef[:, None]

    G_sup = G[support_idx, support_idx[:,None]]

    # DOTS
    dots_tensor = dots_numpy(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb)

    order_0 = order_delta[unique[0]]
    order_1 = order_delta[unique[1]]
    G_tensor = np.einsum('hijk, jk->hijk', dots_tensor, G_sup)
    dot_left = np.einsum('i, hjik', alpha.ravel(), G_tensor)
    dots = np.einsum('jki, i', dot_left, alpha.ravel()) 

    # DISTS
    dots_diag = np.diag(dots)[:, None]
    dists = dots_diag + dots_diag.T - 2*dots

    return dists, dots


def dots_cupy(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb):
    
    T = len(unique)
    n_sv = len(support_idx)

    v_mat = cp.zeros((T, T, n_sv))
    
    for i, tx in enumerate(unique):        
        indX = groups_idx[i]
        order_tx = order_delta[tx]
        for j, ty in enumerate(unique):
            indY = groups_idx[j]                
            order_ty = order_delta[ty]
            vX = cp.zeros((n, 1))
            vX[indX] = delta_inv[order_tx, order_ty]
            v_mat[order_tx, order_ty] = vX[support_idx].flatten()
    v_sum = v_mat.sum(axis=0)


    dots_tensor = (1 - lamb)**2 * cp.einsum('hi,jk->hjik', v_sum, v_sum)

    return dots_tensor

def distances_cupy(G, dual_coef, support_idx, unique, groups_idx, delta_inv, order_delta, lamb, **kwargs):

    T = len(unique)
    n = G.shape[0]
    alpha = dual_coef[:, None]

    G_sup = G[support_idx, support_idx[:,None]]

    # DOTS
    dots_tensor = dots_cupy(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb)

    G_tensor = cp.einsum('hijk, jk->hijk', dots_tensor, G_sup)
    dot_left = cp.einsum('i, hjik', alpha.ravel(), G_tensor)
    dots = cp.einsum('jki, i', dot_left, alpha.ravel()) 

    # DISTS
    dots_diag = np.diag(dots)[:, None]
    dists = dots_diag + dots_diag.T - 2*dots

    return cp.asnumpy(dists), cp.asnumpy(dots)

# @profile()
def distances_divide_cupy(G, dual_coef, support_idx, unique, groups_idx, delta_inv, order_delta, lamb, **kwargs):

    T = len(unique)
    n = G.shape[0]
    alpha = dual_coef[:, None]
    dots = np.zeros([T, T])

    G_sup = G[support_idx, support_idx[:,None]]

    if 'den' in kwargs:
        den = kwargs['den']
    else:
        den = 1

    dots_rowR = np.zeros([1, T])

    vsum = get_vsum(n, unique, groups_idx, support_idx, delta_inv, order_delta, lamb)

    for i, tR in enumerate(unique):
        order_R = order_delta[tR]
        #steps = list(zip(range(0, T, step), range(step, T, step)))
        # ic(steps)
        while True: # Loop to slice the rows smaller and smaller so cupy can allocate for cp.einsum('i,jk->jik', vsum[order_R], vsum[start:end])
            step = int(T/den)

            if step == 0:
                raise ValueError('Step is 0')
            
            steps = list(range(0, T, step))
            steps.append(T)
            try:
                for start, end in zip(steps[:-1], steps[1:]):
                    # ic(start, end)                
                    dots_tensor = (1 - lamb)**2 * cp.einsum('i,jk->jik', vsum[order_R], vsum[start:end])
                    G_tensor = cp.einsum('ijk, jk->ijk', dots_tensor, G_sup)
                    dot_left = cp.einsum('i, jik', alpha.ravel(), G_tensor)
                    dots_rowR[0, start:end] = cp.asnumpy(cp.einsum('ji, i', dot_left, alpha.ravel()))
                    # ic(dots_rowR)
            except OutOfMemoryError as e:
                # print(e)
                den *= 2
                # print('Trying with size {}'.format(int(T/den)))
                pass
            else:
                break
        
        dots[order_R, :] = cp.asnumpy(dots_rowR)
                

    # ic(dots)

    # DISTS
    dots_diag = np.diag(dots)[:, None]
    dists = dots_diag + dots_diag.T - 2*dots

    return cp.asnumpy(dists), cp.asnumpy(dots)


def optimize_adjacency_heur(dists, dots, order_delta, nu, mu):
    B = -1/dists
    
    # print('dots')
    # print(self.dots)
    for ti, i in order_delta:
        B[i, i] = 0                
        B[i, :] /= -np.sum(B[i, :])
        B[i, i] = -np.sum(B[i, :])

    return B

def optimize_adjacency_entropy(dists, dots, order_delta, nu, mu):
    # B computation
    num = np.exp(-(nu / mu) * dists) # 1e-16 * np.ones(dists.shape) # añadimos el 0 máquina
    # np.fill_diagonal(num, 0) # Se comenta para añadir la diagonal 
    den = np.sum(num, axis=1)
    B = num/den[:, None]

    # print('optimize_adjacency -----')
    # print(dists)
    # print(B)

    return B

def optimize_adjacency_cos(dists, dots, order_delta, nu, mu):
    T = dists.shape[0]

    B = np.zeros(dists.shape)
    
    for ti, i in order_delta:
        for tj, j in order_delta:
            B[i, j] = 1 - (dists[i, j] / (dots[i, i] + dots[j, j])) 

    return B

def complete_laplacian(B, order_delta):
    # complete L
    L = deepcopy(-B)
    for i in range(len(order_delta)):
        L[i, i] = np.sum(B[i, :]) - B[i, i]

    return L



def get_kernel_fun(kernel):
        # if not isinstance(kernel, (str, types.FunctionType)):
        #     raise Exception('kernel of wrong type')
        if isinstance(kernel, str):
            kernel_f = getattr(pairwise, kernel+'_kernel')
        else:
            kernel_f = kernel
        return kernel_f


def apply_kernel(kernel, x, y, gamma):
        kernel_f = get_kernel_fun(kernel)
        if kernel_f == pairwise.rbf_kernel:
            return kernel_f(x, y, gamma)
        else:
            return kernel_f(x, y)


def get_deltainv(delta, nu, ind_reg=True):
    if ind_reg:
        delta_full = nu * delta + np.identity(delta.shape[0])
    else:
        delta_full = delta
    deltainv = np.linalg.pinv(delta_full)

    return deltainv



def mtl_kernel_graphlap(X, Y, kernel, gamma, deltainv, order_delta, task_info=-1, G=None):
    """
    We create a custom kernel for multitask learning.
        If task_info is a scalar it is assumed to be the column of the task
        If task_info is an array it is assumed to be the task indexes
    """
    task_col = task_info

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
    A_kron = np.zeros((nX, nY))

    
    for i, tx in enumerate(unique_X):
        for j, ty in enumerate(unique_Y):
            indX = groups_idx_X[i]
            indY = groups_idx_Y[j]
            order_tx = order_delta[tx]
            order_ty = order_delta[ty]
            a_xy = deltainv[order_tx, order_ty]
            A_kron[indX[:, None], indY] = a_xy
    if G is None:
        Q = apply_kernel(kernel, X_data, Y_data, gamma)
    else:
        Q = G

    # Tensor product
    Q_graphL = np.multiply(A_kron, Q)
    
    return Q_graphL
