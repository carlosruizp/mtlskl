<!-- # convexMTLPyTorch
Implementation of ConvexMTL neural networks using PyTorch. -->

# MTLSKL

This is a package that implements some multi-task learning (MTL) methods following the scikit-learn style.

It includes the following implementations:
* Convex MTL L1, L2 and LS-SVMs, as explained in [Convex Formulation for Multi-Task L1-, L2-, and LS-SVMs](https://www.scinapse.io/papers/3173400759).
* Adaptive Graph Laplacian L1, L2 and LS-SVMs, as explained in [Adaptive Graph Laplacian for Convex Multi-Task Learning SVM](https://www.scinapse.io/papers/3201347008).

## Installation

You can install mtlskl using pip:
```bash
pip install mtlskl
```

## Usage

Here is a simple example to get you started:

```python
import numpy as np
from mtlskl.data.MTLSyntheticDataset import MTLFunctionsRegression
from sklearn.model_selection import train_test_split

from mtlskl.model.svm.convexmtl.ConvexMTLSVM import ConvexMTLSVR

# Define data 
mtlds = MTLFunctionsRegression()
X, y = mtlds.X, mtlds.y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize the model
estim = ConvexMTLSVR(C=10, lamb=.5)

# Train the model
estim.fit(X_train, y_train)

# Make predictions
pred = estim.predict(X_test)

# Plot result
mtlds.plot_data_functions(X_test, pred)

```


