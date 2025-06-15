// ex1_vector_add.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel for vector addition
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // Calculate global index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

// CPU implementation for verification
void vectorAdd_CPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Function to verify results
bool verifyResults(const float *cpu_results, const float *gpu_results, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_results[i] - gpu_results[i]) > 1e-5) {
            printf("Verification failed at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_results[i], gpu_results[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Define vector size
    const int N = 10000000; // 10M elements for better timing comparison
    const size_t size = N * sizeof(float);
    
    // Host vectors
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size); // GPU results
    float *h_c_cpu = (float*)malloc(size); // CPU results
    
    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // Define kernel launch parameters
    const int THREADS_PER_BLOCK = 256;
    const int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CPU timing
    clock_t cpu_start, cpu_end;
    
    // Run CPU implementation and measure time
    cpu_start = clock();
    vectorAdd_CPU(h_a, h_b, h_c_cpu, N);
    cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // in ms
    
    // Launch kernel and measure time
    cudaEventRecord(start);
    vectorAdd<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool correct = verifyResults(h_c_cpu, h_c, N);
    
    // Print timing information
    printf("Vector Addition (N=%d)\n", N);
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Output for speedup results file
    printf("ex1_vector_add,%.3f,%.3f,%.2f\n", cpu_time, gpu_time, cpu_time / gpu_time);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}