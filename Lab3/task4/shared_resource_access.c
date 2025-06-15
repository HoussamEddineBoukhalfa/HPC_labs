
// task4/shared_resource_access.c
#include <stdio.h>
#include <omp.h>

int main() {
    int counter = 0;

    // Without synchronization
    #pragma omp parallel for
    for (int i = 0; i < 100000; i++) {
        counter++;
    }
    printf("Counter without synchronization: %d\n", counter);

    counter = 0;

    // With synchronization
    #pragma omp parallel for
    for (int i = 0; i < 100000; i++) {
        #pragma omp atomic
        counter++;
    }
    printf("Counter with synchronization: %d\n", counter);

    return 0;
}
