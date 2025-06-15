#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// CUDA kernel for Euclidean distance calculation
__global__ void euclideanDist_kernel(float *batchA, float *batchB, float *distances, 
                                     int num_vectors, int dim) {
    // Calculate the vector index this thread is responsible for
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (vec_idx < num_vectors) {
        float sum_sq = 0.0f;
        
        // Calculate the starting index for this vector pair
        int base_idx = vec_idx * dim;
        
        // Loop through each dimension
        for (int j = 0; j < dim; j++) {
            // Calculate the linear index for the j-th element
            int elem_idx = base_idx + j;
            
            // Calculate the squared difference
            float diff = batchA[elem_idx] - batchB[elem_idx];
            sum_sq += diff * diff;
        }
        
        // Calculate the Euclidean distance and store it
        distances[vec_idx] = sqrtf(sum_sq);
    }
}

// CPU implementation for verification
void euclideanDist_cpu(float *batchA, float *batchB, float *distances, 
                       int num_vectors, int dim) {
    for (int i = 0; i < num_vectors; i++) {
        float sum_sq = 0.0f;
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            float diff = batchA[idx] - batchB[idx];
            sum_sq += diff * diff;
        }
        distances[i] = sqrtf(sum_sq);
    }
}

int main() {
    // Set problem parameters
    int num_vectors = 1000000;  // Number of vector pairs
    int dim = 128;             // Dimension of each vector
    
    // Allocate host memory
    size_t batch_size = num_vectors * dim * sizeof(float);
    size_t dist_size = num_vectors * sizeof(float);
    
    float *h_batchA = (float*)malloc(batch_size);
    float *h_batchB = (float*)malloc(batch_size);
    float *h_distances = (float*)malloc(dist_size);
    float *h_distances_cpu = (float*)malloc(dist_size);
    
    // Initialize input batches with random data
    srand(42);
    for (int i = 0; i < num_vectors * dim; i++) {
        h_batchA[i] = ((float)rand() / RAND_MAX) * 10.0f;
        h_batchB[i] = ((float)rand() / RAND_MAX) * 10.0f;
    }
    
    // Allocate device memory
    float *d_batchA, *d_batchB, *d_distances;
    cudaMalloc(&d_batchA, batch_size);
    cudaMalloc(&d_batchB, batch_size);
    cudaMalloc(&d_distances, dist_size);
    
    // Copy input data to device
    cudaMemcpy(d_batchA, h_batchA, batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_batchB, h_batchB, batch_size, cudaMemcpyHostToDevice);
    
    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_vectors + threadsPerBlock - 1) / threadsPerBlock;
    
    // CPU timing
    clock_t cpu_start = clock();
    euclideanDist_cpu(h_batchA, h_batchB, h_distances_cpu, num_vectors, dim);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC; // ms
    
    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    euclideanDist_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_batchA, d_batchB, d_distances, num_vectors, dim);
    cudaEventRecord(stop);
    
    // Copy result back to host
    cudaMemcpy(h_distances, d_distances, dist_size, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    
    // Verify results
    bool pass = true;
    for (int i = 0; i < num_vectors; i++) {
        if (fabsf(h_distances[i] - h_distances_cpu[i]) > 1e-5f) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", 
                   i, h_distances[i], h_distances_cpu[i]);
            pass = false;
            break;
        }
    }
    
    // Print detailed output
    printf("Batched Euclidean Distance (Vectors=%d, Dim=%d)\n", num_vectors, dim);
    // Sample results for first few vectors
    printf("Sample results (first 3 vectors):\n");
    for (int i = 0; i < 3 && i < num_vectors; i++) {
        printf("Vector %d: CPU=%.6f, GPU=%.6f\n", i, h_distances_cpu[i], h_distances[i]);
    }
    printf("CPU execution time: %.3f ms\n", cpu_time);
    printf("GPU execution time: %.3f ms\n", gpu_time_ms);
    printf("Speedup: %.2fx\n\n", cpu_time / gpu_time_ms);
    
    // Print CSV format for run_all_tests.sh
    if (pass) {
        printf("ex6_euclidean_dist,%.3f,%.3f,%.3f\n", cpu_time, gpu_time_ms, cpu_time / gpu_time_ms);
    } else {
        printf("ex6_euclidean_dist,%.3f,%.3f,FAILED\n", cpu_time, gpu_time_ms);
    }
    
    // Clean up
    cudaFree(d_batchA);
    cudaFree(d_batchB);
    cudaFree(d_distances);
    free(h_batchA);
    free(h_batchB);
    free(h_distances);
    free(h_distances_cpu);
    
    return 0;
}