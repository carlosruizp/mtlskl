from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, r2_score, make_scorer
from sklearn.metrics import mean_absolute_error as mae
from sklearn.base import BaseEstimator
import numpy as np
import numpy_indexed as npi
from copy import deepcopy
from experiment.TaskDataHandler import getDataTask
import types

def create_itl_scoring(scoring_fun, greater_is_better=True):
    def score_itl_template(y_true, y_pred):
        NaN_mask = np.isnan(y_pred)
        if greater_is_better:
            return scoring_fun(y_true[~NaN_mask], y_pred[~NaN_mask])
        else:
            return -scoring_fun(y_true[~NaN_mask], y_pred[~NaN_mask])
    return score_itl_template


def neg_r2_score(y_true, y_pred, sample_weight=None):
    return -r2_score(y_true, y_pred, sample_weight)


class ITLContainer(BaseEstimator):

    """docstring for itlSVM."""
    def __init__(self, models_dic=None, task_info=None, iso_task=None, score_fun=None):
        super().__init__()
        self.task_info = task_info
        self.iso_task = iso_task
        self.models_dic = models_dic
        self.score_fun=None

    '''
        Supongo que en train me pasan todas las tareas para poder hacer
        el diccionario completo.
    '''
    def fit(self, X, y, task_info=-1, sample_weight=None):
        n, m = X.shape
        self.task_info = task_info
        task_col = task_info
        data_columns = np.delete(range(m), task_col)
        self.X_train = X
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self.y = y_2d
        # groups
        self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        groups_dic = dict(zip(self.unique, self.groups_idx))

        # print(X.shape, y.shape)
        # print('Task:', self.iso_task)

        if self.iso_task is None:
            for i, (k, model) in enumerate(self.models_dic.items()):
                if len(self.groups_idx[i]) == 1:
                    row_idx = self.groups_idx[i]
                    X_g = X[row_idx, data_columns].reshape(1, -1)
                else:
                    row_idx = self.groups_idx[i][:, None]
                    X_g = X[row_idx, data_columns]
                y_g = self.y[self.groups_idx[i]]
                model.fit(X_g, y_g.ravel())
        else:
            if len(groups_dic[self.iso_task]) == 1:
                row_idx = groups_dic[self.iso_task]
                X_g = X[row_idx, data_columns].reshape(1, -1)
            else:
                row_idx = groups_dic[self.iso_task][:, None]
                X_g = X[row_idx, data_columns]
            y_g = self.y[groups_dic[self.iso_task]]
            # print(X_g.shape, y_g.shape)
            self.models_dic[self.iso_task].fit(X_g, y_g.ravel())
            self.model = self.models_dic[self.iso_task]
        # print('Score:', mae(y_g, self.model.predict(X_g)))


    def predict(self, X, task_info=None):
        n, m = X.shape
        if task_info is None:
            task_info = self.task_info
        task_col = task_info
        data_columns = np.delete(range(m), task_col)
        unique, groups_idx = npi.group_by(X[:, task_col],
                                          np.arange(n))

        self.predictions = {}
        if self.iso_task is not None:
            for i, t in enumerate(unique):
                if self.iso_task == t:
                    model = self.models_dic[t]
                    if len(groups_idx[i]) == 1:
                        row_idx = groups_idx[i]
                        X_g = X[row_idx, data_columns].reshape(1, -1)
                    else:
                        row_idx = groups_idx[i][:, None]
                        X_g = X[row_idx, data_columns]
                    pred = model.predict(X_g)
                    self.predictions[t] = deepcopy(pred)
                else:
                    self.predictions[t] =  np.empty(len(groups_idx[i]))
                    self.predictions[t][:] = np.NaN
        else:
            print(unique)
            print(self.models_dic)
            for i, t in enumerate(unique):
                # Probamos por si es float
                if t in self.models_dic:
                    model = self.models_dic[t]
                else:
                    model = self.models_dic[float(t)]
                if len(groups_idx[i]) == 1:
                    row_idx = groups_idx[i]
                    X_g = X[row_idx, data_columns].reshape(1, -1)
                else:
                    row_idx = groups_idx[i][:, None]
                    X_g = X[row_idx, data_columns]
                pred = model.predict(X_g)
                self.predictions[t] = deepcopy(pred)
        order = np.argsort(np.concatenate(groups_idx))
        return np.concatenate(list(self.predictions.values()))[order]

    def score(self, X, y, sample_weight=None, task_info=None):
        if self.score_fun is None:
            raise Exception('object has not been given a score_fun')
        n, m = X.shape
        if task_info is None:
            task_info = self.task_info
        task_col = task_info
        data_columns = np.delete(range(m), task_col)
        unique, groups_idx = npi.group_by(X[:, task_col],
                                          np.arange(n))
        self.predictions = {}
        self.scores = {}
        for i, t in enumerate(unique):
            model = self.models_dic[t]
            if len(groups_idx[i]) == 1:
                row_idx = groups_idx[i]
                X_g = X[row_idx, data_columns].reshape(1, -1)
            else:
                row_idx = groups_idx[i][:, None]
                X_g = X[row_idx, data_columns]
            y_g = y[groups_idx[i]]
            pred = model.predict(X_g)
            self.predictions[t] = deepcopy(pred)
            score = self.score_fun(y_g, pred)
            self.scores[t] = score

        order = np.concatenate(groups_idx)
        if self.iso_task is None:
            return self.score_fun(y[order],
                                np.concatenate(list(self.predictions.values())))
        else:
            return self.scores[self.iso_task]
