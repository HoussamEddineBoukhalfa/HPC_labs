#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

// CUDA kernel for partial argmax
__global__ void argmax_partial_kernel(float *input, int n, 
                                     float *max_vals_partial, 
                                     int *max_idxs_partial) {
    extern __shared__ float s_mem[];
    float *s_vals = s_mem;
    int *s_idxs = (int*)&s_vals[blockDim.x];
    
    int tid = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory with boundary check
    if (gidx < n) {
        s_vals[tid] = input[gidx];
        s_idxs[tid] = gidx;
    } else {
        s_vals[tid] = -FLT_MAX;
        s_idxs[tid] = -1;
    }
    
    __syncthreads();
    
    // Perform parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes the result for this block
    if (tid == 0) {
        max_vals_partial[blockIdx.x] = s_vals[0];
        max_idxs_partial[blockIdx.x] = s_idxs[0];
    }
}

// CPU implementation for verification
int argmax_cpu(float *input, int n) {
    float max_val = input[0];
    int max_idx = 0;
    
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

int main() {
    // Set problem parameters
    int n = 10000000;  // Size of input vector
    
    // Allocate host memory
    float *h_input = (float*)malloc(n * sizeof(float));
    
    // Initialize input with random data
    srand(42);
    for (int i = 0; i < n; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 100.0f;
    }
    // Make one element clearly the maximum (for easier verification)
    int true_max_idx = rand() % n;
    h_input[true_max_idx] = 1000.0f;
    
    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory for partial results
    float *d_max_vals_partial;
    int *d_max_idxs_partial;
    cudaMalloc(&d_max_vals_partial, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_max_idxs_partial, blocksPerGrid * sizeof(int));
    
    float *h_max_vals_partial = (float*)malloc(blocksPerGrid * sizeof(float));
    int *h_max_idxs_partial = (int*)malloc(blocksPerGrid * sizeof(int));
    
    // CPU timing
    clock_t cpu_start = clock();
    int cpu_max_idx = argmax_cpu(h_input, n);
    float cpu_max_val = h_input[cpu_max_idx];
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // ms
    
    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Launch kernel with shared memory for values and indices
    size_t shared_mem_size = threadsPerBlock * sizeof(float) + threadsPerBlock * sizeof(int);
    argmax_partial_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
        d_input, n, d_max_vals_partial, d_max_idxs_partial);
    
    cudaEventRecord(stop);
    
    // Copy partial results back to host
    cudaMemcpy(h_max_vals_partial, d_max_vals_partial, 
               blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_idxs_partial, d_max_idxs_partial, 
               blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    
    // Final reduction step on CPU
    float global_max_val = h_max_vals_partial[0];
    int global_max_idx = h_max_idxs_partial[0];
    
    for (int i = 1; i < blocksPerGrid; i++) {
        if (h_max_vals_partial[i] > global_max_val) {
            global_max_val = h_max_vals_partial[i];
            global_max_idx = h_max_idxs_partial[i];
        }
    }
    
    // Verify results
    bool pass = (global_max_idx == cpu_max_idx);
    
    // Print detailed output
    printf("Parallel Partial Argmax (N=%d)\n", n);
    printf("CPU result: max value %.6f at index %d\n", cpu_max_val, cpu_max_idx);
    printf("GPU result: max value %.6f at index %d\n", global_max_val, global_max_idx);
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time_ms);
    printf("Speedup: %.2fx\n\n", cpu_time / gpu_time_ms);
    
    // Print CSV format for run_all_tests.sh
    if (pass) {
        printf("ex7_argmax,%.3f,%.3f,%.3f\n", cpu_time, gpu_time_ms, cpu_time / gpu_time_ms);
    } else {
        printf("ex7_argmax,%.3f,%.3f,FAILED (GPU: %d, CPU: %d)\n", 
               cpu_time, gpu_time_ms, global_max_idx, cpu_max_idx);
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_max_vals_partial);
    cudaFree(d_max_idxs_partial);
    free(h_input);
    free(h_max_vals_partial);
    free(h_max_idxs_partial);
    
    return 0;
}