from sklearn.svm import SVC

class MILSVC:
    """
    A class for support vector machine (SVM) classification using scikit-learn.
    
    Parameters
    ----------
    kernel : str, default: 'rbf'
        The kernel to use in the SVM. Can be 'linear', 'poly', 'rbf', 'sigmoid', 
        'precomputed' or a callable.
    
    C : float, default: 1.0
        Regularization parameter. The strength of the regularization is inversely
        proportional to C. Must be strictly positive.
    
    gamma : str, default: 'scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'scale'
        (default) then 1 / (n_features * X.var()) is used instead.
    
    degree : int, default: 3
        Degree of the polynomial kernel function ('poly'). Ignored by all other
        kernels.
    
    coef0 : float, default: 0.0
        Independent term in kernel function. It is only significant in 'poly' and
        'sigmoid'.
    
    Attributes
    ----------
    model : sklearn.svm.SVC
        The underlying scikit-learn SVC model object
    
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
        
    def fit(self, X, y):
        """
        Fit the SVM model to the provided training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns the instance of the class
        
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to predict the class labels for
        
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for each data sample
        
        """
        y_pred = self.model.predict(X)
        return y_pred
    
    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True labels for X
        
        Returns
        -------
        score : float
            Mean accuracy
        """
        score_value = self.model.score(X, y)
        return score_value
