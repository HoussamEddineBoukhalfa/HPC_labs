
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int world_rank, world_size;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Hello from process %d of %d processes!\n", world_rank, world_size);

    MPI_Finalize();
    return 0;
}
