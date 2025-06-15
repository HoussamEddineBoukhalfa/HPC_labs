// ex4_dot_product.cu
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

// Kernel for vector dot product with shared memory
__global__ void vectorDotProduct_partial(const float *a, const float *b, float *partial_c, int n) {
    __shared__ float cache[256]; // Assuming block size is 256
    
    int tid = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize cache value
    float temp = 0.0f;
    if (gidx < n) {
        temp = a[gidx] * b[gidx];
    }
    cache[tid] = temp;
    
    // Synchronize threads in block
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        partial_c[blockIdx.x] = cache[0];
    }
}

// CPU implementation for verification
float dotProduct_CPU(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    // Define vector size
    const int N = 8000000; // 8M elements for better timing comparison
    const size_t vector_size = N * sizeof(float);
    
    // Host vectors
    float *h_a = (float*)malloc(vector_size);
    float *h_b = (float*)malloc(vector_size);
    
    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Device vectors
    float *d_a, *d_b;
    
    // Allocate device memory for vectors
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, vector_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, vector_size));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, vector_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, vector_size, cudaMemcpyHostToDevice));
    
    // Define kernel launch parameters
    const int THREADS_PER_BLOCK = 256;
    const int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Allocate memory for partial results
    const size_t partial_size = numBlocks * sizeof(float);
    float *h_partial_c = (float*)malloc(partial_size);
    float *d_partial_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_partial_c, partial_size));
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CPU timing
    clock_t cpu_start, cpu_end;
    
    // Run CPU implementation and measure time
    cpu_start = clock();
    float cpu_result = dotProduct_CPU(h_a, h_b, N);
    cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // in ms
    
    // Launch kernel and measure time
    cudaEventRecord(start);
    vectorDotProduct_partial<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_partial_c, N);
    cudaEventRecord(stop);
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    cudaEventSynchronize(stop);
    
    // Calculate elapsed kernel time
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    
    // Copy partial results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_partial_c, d_partial_c, partial_size, cudaMemcpyDeviceToHost));
    
    // Compute final result on host by summing partial results
    float gpu_result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        gpu_result += h_partial_c[i];
    }
    
    // Compute total GPU time (kernel + final reduction)
    float gpu_time = kernel_time;
    
    // Verify results
    bool correct = fabs(cpu_result - gpu_result) < 1e-5 * N;
    
    // Print timing information
    printf("Vector Dot Product (N=%d)\n", N);
    printf("CPU result: %f\n", cpu_result);
    printf("GPU result: %f\n", gpu_result);
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Output for speedup results file
    printf("ex4_dot_product,%.3f,%.3f,%.2f\n", cpu_time, gpu_time, cpu_time / gpu_time);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_partial_c);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}