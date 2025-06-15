
// task3/parallel_matrix_multiplication.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 500

int main() {
    double a[N][N], b[N][N], c[N][N];

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i][j] = i + j;
            b[i][j] = i - j;
            c[i][j] = 0;
        }

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                c[i][j] += a[i][k] * b[k][j];
    double end = omp_get_wtime();

    printf("Matrix multiplication done in %f seconds\n", end - start);
    return 0;
}
