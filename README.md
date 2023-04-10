<!-- # convexMTLPyTorch
Implementation of ConvexMTL neural networks using PyTorch. -->

# MTLSKL

This is a package that implements the multi-task learning (MTL) methods presented in [[1]](#1), [[2]](#2), [[3]](#3) and [[4]](#4) following the scikit-learn style.

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

# Cite our work
If you use this implementation, please cite the corresponding papers, which are shown below.

# Papers
<a id="1">[1]</a> 
Carlos Ruiz and Carlos M. Alaíz and José R. Dorronsoro (2019). 
Convex Graph Laplacian Multi-Task Learning SVM. 
HAIS Proceedings. Vol. 1173, Springer, 404-415.

<a id="2">[2]</a> 
Carlos Ruiz and Carlos M. Alaíz and José R. Dorronsoro (2020). 
Convex Graph Laplacian Multi-Task Learning SVM.
ICANN. Vol. 12397. Springer, 142-154.

<a id="3">[3]</a>
Carlos Ruiz and Carlos M. Alaíz and José R. Dorronsoro (2021). 
Convex formulation for multi-task L1-, L2-, and LS-SVMs.
Neurocomputing 456, 404-415.

<a id="4">[4]</a> 
Carlos Ruiz and Carlos M. Alaíz and José R. Dorronsoro (2021). 
Adaptive Graph Laplacian for Convex Multi-Task Learning SVM.
HAIS Proceedings. Vol. 12886, Springer, 219-230.



