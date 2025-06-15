#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel for matrix addition using 2D blocks and threads
__global__ void matrixAdd(float *A, float *B, float *C, int width, int height) {
    // Calculate the global column and row indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within matrix bounds
    if (col < width && row < height) {
        // Calculate linear index for row-major storage
        int index = row * width + col;
        
        // Perform the addition
        C[index] = A[index] + B[index];
    }
}

// CPU implementation for verification
void matrixAdd_cpu(float *A, float *B, float *C, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = row * width + col;
            C[index] = A[index] + B[index];
        }
    }
}

int main() {
    // Set matrix dimensions
    int width = 4096;   // Width of the matrix
    int height = 4096;  // Height of the matrix
    
    size_t matrix_size = width * height * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(matrix_size);
    float *h_B = (float*)malloc(matrix_size);
    float *h_C = (float*)malloc(matrix_size);
    float *h_C_cpu = (float*)malloc(matrix_size);
    
    // Initialize matrices with random data
    srand(42);
    for (int i = 0; i < width * height; i++) {
        h_A[i] = ((float)rand() / RAND_MAX) * 10.0f;
        h_B[i] = ((float)rand() / RAND_MAX) * 10.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrix_size);
    cudaMalloc(&d_B, matrix_size);
    cudaMalloc(&d_C, matrix_size);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
    
    // Define 2D block dimensions
    dim3 threadsPerBlock(16, 16);
    
    // Calculate 2D grid dimensions with ceiling division
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    // CPU timing
    clock_t cpu_start = clock();
    matrixAdd_cpu(h_A, h_B, h_C_cpu, width, height);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // ms
    
    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);
    cudaEventRecord(stop);
    
    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    
    // Verify results
    bool pass = true;
    for (int i = 0; i < width * height; i++) {
        if (fabsf(h_C[i] - h_C_cpu[i]) > 1e-5f) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", 
                   i, h_C[i], h_C_cpu[i]);
            pass = false;
            break;
        }
    }
    
    // Print detailed output
    printf("Matrix Addition (Width=%d, Height=%d)\n", width, height);
    printf("Sample results (first element): A[0]=%f + B[0]=%f = C[0]=%f\n", 
           h_A[0], h_B[0], h_C[0]);
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time_ms);
    printf("Speedup: %.2fx\n\n", cpu_time / gpu_time_ms);
    
    // Print CSV format for run_all_tests.sh
    if (pass) {
        printf("ex8_matrix_add,%.3f,%.3f,%.3f\n", 
               cpu_time, gpu_time_ms, cpu_time / gpu_time_ms);
    } else {
        printf("ex8_matrix_add,%.3f,%.3f,FAILED\n", cpu_time, gpu_time_ms);
    }
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    
    return 0;
}