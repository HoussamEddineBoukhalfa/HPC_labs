// ex3_mat_vec_mul.cu
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

// Kernel for matrix-vector multiplication
__global__ void matVecMul_kernel(const float *M, const float *x, float *y, int rows, int cols) {
    // Calculate row index (global thread index)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < rows) {
        float sum = 0.0f;
        
        // Compute dot product of row and vector x
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col; // Row-major index
            sum += M[idx] * x[col];
        }
        
        // Store result
        y[row] = sum;
    }
}

// CPU implementation for verification
void matVecMul_CPU(const float *M, const float *x, float *y, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += M[row * cols + col] * x[col];
        }
        y[row] = sum;
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
    // Define matrix and vector dimensions
    const int ROWS = 4000;
    const int COLS = 4000;
    
    const size_t matrix_size = ROWS * COLS * sizeof(float);
    const size_t vector_size = COLS * sizeof(float);
    const size_t result_size = ROWS * sizeof(float);
    
    // Host memory allocation
    float *h_M = (float*)malloc(matrix_size);
    float *h_x = (float*)malloc(vector_size);
    float *h_y = (float*)malloc(result_size); // GPU results
    float *h_y_cpu = (float*)malloc(result_size); // CPU results
    
    // Initialize host data
    for (int i = 0; i < ROWS * COLS; i++) {
        h_M[i] = rand() / (float)RAND_MAX;
    }
    
    for (int i = 0; i < COLS; i++) {
        h_x[i] = rand() / (float)RAND_MAX;
    }
    
    // Device memory allocation
    float *d_M, *d_x, *d_y;
    CHECK_CUDA_ERROR(cudaMalloc(&d_M, matrix_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, vector_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, result_size));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_M, h_M, matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, vector_size, cudaMemcpyHostToDevice));
    
    // Define kernel launch parameters
    const int THREADS_PER_BLOCK = 256;
    const int numBlocks = (ROWS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CPU timing
    clock_t cpu_start, cpu_end;
    
    // Run CPU implementation and measure time
    cpu_start = clock();
    matVecMul_CPU(h_M, h_x, h_y_cpu, ROWS, COLS);
    cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // in ms
    
    // Launch kernel and measure time
    cudaEventRecord(start);
    matVecMul_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_M, d_x, d_y, ROWS, COLS);
    cudaEventRecord(stop);
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y, result_size, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool correct = verifyResults(h_y_cpu, h_y, ROWS);
    
    // Print timing information
    printf("Matrix-Vector Multiplication (%dx%d)\n", ROWS, COLS);
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Output for speedup results file
    printf("ex3_mat_vec_mul,%.3f,%.3f,%.2f\n", cpu_time, gpu_time, cpu_time / gpu_time);
    
    // Free device memory
    cudaFree(d_M);
    cudaFree(d_x);
    cudaFree(d_y);
    
    // Free host memory
    free(h_M);
    free(h_x);
    free(h_y);
    free(h_y_cpu);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}