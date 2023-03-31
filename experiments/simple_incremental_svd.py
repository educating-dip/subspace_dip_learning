from typing import Optional, Any, Union
import scipy
import random
import numpy as np
import tensorly as tl
from subspace_dip.dip import IncremetalSVD

seed = 1
random.seed(seed)
np.random.seed(seed)
matrix = np.random.randn(1000, 1000)
matrix[np.diag_indices(matrix.shape[0])] += np.float32(
    np.exp(-np.random.uniform(low=0, high=10, size=1000)))
matrix = matrix[:, :100]

U, s, VT = tl.partial_svd(matrix=matrix, n_eigenvecs=100)
svd = IncremetalSVD(n_eigenvecs=100, batch_size=1, gamma=1)
svd.start_tracking(data=matrix[:, :1])

for column in range(1, 100):
    stop = svd.update(C=matrix[:, column])
    if stop: 
        break
print(scipy.linalg.norm(s - svd.s) / scipy.linalg.norm(s))
vects = np.random.uniform(low=0, high=100, size=(1000, 100))
xs_inc = []
for a in vects:
    x = svd.U @ np.diag(svd.s) @ svd.V.T @ a
    xs_inc.append(x)
print(np.mean(xs_inc))

xs_svd = []
for a in vects:
    x = U @ np.diag(s) @ VT @ a
    xs_svd.append(x)
print(np.mean(xs_svd))

xs_exact = []
for a in vects:
    x = matrix @ a
    xs_exact.append(x)
print(np.mean(xs_exact))






