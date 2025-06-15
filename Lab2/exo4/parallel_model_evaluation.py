
# exo4/parallel_model_evaluation.py
from mpi4py import MPI
import numpy as np
from sklearn.metrics import mean_squared_error

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    X_test = np.random.rand(100, 1)
    y_test = 3 * X_test.squeeze() + 2 + np.random.randn(100) * 0.1
    X_splits = np.array_split(X_test, size)
    y_splits = np.array_split(y_test, size)
    for i in range(1, size):
        comm.send((X_splits[i], y_splits[i]), dest=i)
    X_local, y_local = X_splits[0], y_splits[0]
else:
    X_local, y_local = comm.recv(source=0)

# Dummy model: y = 3x + 2
y_pred = 3 * X_local.squeeze() + 2
local_mse = mean_squared_error(y_local, y_pred)

comm.Reduce(np.array(local_mse), np.array(local_mse) if rank == 0 else None, op=MPI.SUM, root=0)

if rank == 0:
    avg_mse = local_mse / size
    print(f"Average MSE from all processes: {avg_mse}")
