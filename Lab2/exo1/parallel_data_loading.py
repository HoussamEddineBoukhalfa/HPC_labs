
# exo1/parallel_data_loading.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.random.rand(100, 10)
    print("Master is loading the data.")
else:
    data = None

data = comm.bcast(data, root=0)

if rank != 0:
    print(f"Worker {rank} received data with shape: {data.shape}")
