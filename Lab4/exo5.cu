// ex5_stencil.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Stencil radius and block size
#define RADIUS 3
#define BLOCK_SIZE 256

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

// Kernel for 1D stencil computation
__global__ void stencil_1d(const float *in, float *out, int n) {
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];
    
    // Calculate global and local indices
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lindex = threadIdx.x + RADIUS;
    
    // Load main element into shared memory
    if (gindex < n) {
        temp[lindex] = in[gindex];
    } else {
        temp[lindex] = 0.0f;
    }
    
    // Load left halo elements
    if (threadIdx.x < RADIUS) {
        if (gindex >= RADIUS) {
            temp[lindex - RADIUS] = in[gindex - RADIUS];
        } else {
            temp[lindex - RADIUS] = 0.0f; // Out of bounds
        }
    }
    
    // Load right halo elements
    if (threadIdx.x >= blockDim.x - RADIUS) {
        if (gindex + RADIUS < n) {
            temp[lindex + RADIUS] = in[gindex + RADIUS];
        } else {
            temp[lindex + RADIUS] = 0.0f; // Out of bounds
        }
    }
    
    // Ensure all data is loaded
    __syncthreads();
    
    // Compute stencil
    if (gindex < n) {
        float result = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += temp[lindex + offset];
        }
        out[gindex] = result;
    }
}

// CPU implementation for verification
void stencil_1d_CPU(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float result = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            int j = i + offset;
            if (j >= 0 && j < n) {
                result += in[j];
            }
        }
        out[i] = result;
    }
}

// Function to verify results
bool verifyResults(const float *cpu_results, const float *gpu_results, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_results[i] - gpu_results[i]) > 1e-5f) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_results[i], gpu_results[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Set problem size
    int n = 10000000;  // Size of the array
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float *h_in = (float*)malloc(bytes);
    float *h_out_cpu = (float*)malloc(bytes);
    float *h_out_gpu = (float*)malloc(bytes);
    
    // Initialize input array with random data
    srand(42);
    for (int i = 0; i < n; i++) {
        h_in[i] = ((float)rand() / RAND_MAX) * 10.0f;
    }
    
    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, bytes));
    
    // Copy input array to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    // Define execution configuration
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // CPU timing
    clock_t cpu_start = clock();
    stencil_1d_CPU(h_in, h_out_cpu, n);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // ms
    
    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    stencil_1d<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    
    // Verify results
    bool pass = verifyResults(h_out_cpu, h_out_gpu, n);
    
    // Print detailed output
    printf("1D Stencil Computation (N=%d, Radius=%d)\n", n, RADIUS);
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time_ms);
    printf("Speedup: %.2fx\n\n", cpu_time / gpu_time_ms);
    
    // Print CSV format for run_all_tests.sh
    if (pass) {
        printf("ex5_stencil,%.3f,%.3f,%.3f\n", cpu_time, gpu_time_ms, cpu_time / gpu_time_ms);
    } else {
        printf("ex5_stencil,%.3f,%.3f,FAILED\n", cpu_time, gpu_time_ms);
    }
    
    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    
    return 0;
}