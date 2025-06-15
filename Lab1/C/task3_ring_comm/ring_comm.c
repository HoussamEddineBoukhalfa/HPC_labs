
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size, message;
    MPI_Status status;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == 0) {
        message = 100;
        MPI_Send(&message, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(&message, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        printf("Process 0 received message %d from process %d\n", message, prev);
    } else {
        MPI_Recv(&message, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        message += rank;
        MPI_Send(&message, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        printf("Process %d received and sent message: %d\n", rank, message);
    }

    MPI_Finalize();
    return 0;
}
