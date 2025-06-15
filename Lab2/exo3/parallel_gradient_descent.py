
# exo3/parallel_gradient_descent.py
from mpi4py import MPI
from sklearn.datasets import make_regression
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    w, b = np.random.rand(1), np.random.rand(1)
    X_splits = np.array_split(X, size)
    y_splits = np.array_split(y, size)
    for i in range(1, size):
        comm.send((X_splits[i], y_splits[i]), dest=i)
    X_local, y_local = X_splits[0], y_splits[0]
else:
    X_local, y_local = comm.recv(source=0)
    w, b = None, None

learning_rate = 0.01
for _ in range(100):
    w = comm.bcast(w, root=0)
    b = comm.bcast(b, root=0)
    y_pred = X_local.dot(w) + b
    grad_w = -2 * np.mean((y_local - y_pred) * X_local)
    grad_b = -2 * np.mean(y_local - y_pred)
    grads = np.array([grad_w, grad_b])
    comm.Reduce(grads, grads if rank == 0 else None, op=MPI.SUM, root=0)
    if rank == 0:
        w -= learning_rate * grads[0] / size
        b -= learning_rate * grads[1] / size
