// ex2_relu.cu
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

// Kernel for ReLU activation function
__global__ void relu_kernel(const float *input, float *output, int n) {
    // Calculate global index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (index < n) {
        output[index] = fmaxf(0.0f, input[index]);
    }
}

// CPU implementation for verification
void relu_CPU(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = fmaxf(0.0f, input[i]);
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
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size); // GPU results
    float *h_output_cpu = (float*)malloc(size); // CPU results
    
    // Initialize host input with both positive and negative values
    for (int i = 0; i < N; i++) {
        h_input[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f; // Values between -1 and 1
    }
    
    // Device vectors
    float *d_input, *d_output;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
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
    relu_CPU(h_input, h_output_cpu, N);
    cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // in ms
    
    // Launch kernel and measure time
    cudaEventRecord(start);
    relu_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool correct = verifyResults(h_output_cpu, h_output, N);
    
    // Print timing information
    printf("ReLU Activation (N=%d)\n", N);
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Output for speedup results file
    printf("ex2_relu,%.3f,%.3f,%.2f\n", cpu_time, gpu_time, cpu_time / gpu_time);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Free host memory
    free(h_input);
    free(h_output);
    free(h_output_cpu);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}