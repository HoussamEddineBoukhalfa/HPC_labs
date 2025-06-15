
# exo2/parallel_kmeans.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.random.rand(100, 2)  # Simplified 2D data
k = 3
centroids = np.random.rand(k, 2) if rank == 0 else None

data = comm.bcast(data, root=0)

for _ in range(5):
    centroids = comm.bcast(centroids, root=0)
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    if rank != 0:
        print(f"Worker {rank} cluster assignments (first 10): {clusters[:10]}")
