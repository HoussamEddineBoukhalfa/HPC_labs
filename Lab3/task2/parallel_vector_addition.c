
// task2/parallel_vector_addition.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 100000000

int main() {
    float *a = (float*)malloc(N * sizeof(float));
    float *b = (float*)malloc(N * sizeof(float));
    float *c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0;
        b[i] = (N - i) * 1.0;
    }

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    double end = omp_get_wtime();

    printf("Vector addition done in %f seconds\n", end - start);

    free(a);
    free(b);
    free(c);
    return 0;
}
