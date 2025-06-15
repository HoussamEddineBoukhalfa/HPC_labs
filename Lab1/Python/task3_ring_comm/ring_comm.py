
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

next_rank = (rank + 1) % size
prev_rank = (rank - 1 + size) % size

if rank == 0:
    message = 100
    comm.send(message, dest=next_rank, tag=0)
    received = comm.recv(source=prev_rank, tag=0)
    print(f"Process 0 received message {received} from process {prev_rank}")
else:
    received = comm.recv(source=prev_rank, tag=0)
    message = received + rank
    comm.send(message, dest=next_rank, tag=0)
    print(f"Process {rank} received {received} and sent {message}")
