#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define checkCudaErrors(call)                                  \
   {                                                           \
      cudaError_t err = call;                                  \
      if (err != cudaSuccess)                                  \
      {                                                        \
         fprintf(stderr, "CUDA error in %s:%d: %s\n",          \
                 __FILE__, __LINE__, cudaGetErrorString(err)); \
         exit(EXIT_FAILURE);                                   \
      }                                                        \
   }

// Sequential implementation of prefix sum
void sequential_scan(const int *input, int *output, int n)
{
   output[0] = input[0];
   for (int i = 1; i < n; i++)
   {
      output[i] = output[i - 1] + input[i];
   }
}

// Naive CUDA kernel for prefix sum (O(n log n) work)
__global__ void naive_scan_kernel(int *input, int *output, int n)
{
   extern __shared__ int temp[];

   int tid = threadIdx.x;
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   // Load input into shared memory
   if (idx < n)
   {
      temp[tid] = input[idx];
   }
   else
   {
      temp[tid] = 0;
   }

   __syncthreads();

   // Perform scan with O(n log n) work
   for (int stride = 1; stride < blockDim.x; stride *= 2)
   {
      int index = (tid + 1) * 2 * stride - 1;
      if (index < blockDim.x)
      {
         temp[index] += temp[index - stride];
      }
      __syncthreads();
   }

   // Down-sweep phase (distribute sums)
   for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
   {
      int index = (tid + 1) * 2 * stride - 1;
      if ((index + stride) < blockDim.x)
      {
         temp[index + stride] += temp[index];
      }
      __syncthreads();
   }

   // Write results to global memory
   if (idx < n)
   {
      output[idx] = temp[tid];
   }
}

// Optimized CUDA kernel for prefix sum (work-efficient)
// This implements the Blelloch scan algorithm
__global__ void work_efficient_scan_kernel(int *input, int *output, int n)
{
   extern __shared__ int temp[];

   int tid = threadIdx.x;
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   // Load input into shared memory
   if (idx < n)
   {
      temp[tid] = input[idx];
   }
   else
   {
      temp[tid] = 0;
   }

   __syncthreads();

   // Reduction phase (up-sweep)
   for (int stride = 1; stride < blockDim.x; stride *= 2)
   {
      int index = (tid + 1) * 2 * stride - 1;
      if (index < blockDim.x)
      {
         temp[index] += temp[index - stride];
      }
      __syncthreads();
   }

   // Clear the last element (for exclusive scan, can be skipped for inclusive)
   if (tid == 0)
   {
      temp[blockDim.x - 1] = 0;
   }

   __syncthreads();

   // Distribution phase (down-sweep)
   for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
   {
      int index = (tid + 1) * 2 * stride - 1;
      if (index < blockDim.x)
      {
         int tmp = temp[index];
         temp[index] += temp[index - stride];
         temp[index - stride] = tmp;
      }
      __syncthreads();
   }

   // Write results to global memory (shift by 1 to make inclusive)
   if (idx < n)
   {
      if (tid > 0)
      {
         output[idx] = temp[tid - 1] + input[idx];
      }
      else
      {
         output[idx] = input[idx]; // First element is just itself
      }
   }
}

// Multi-block scan for large arrays
__global__ void block_sum_kernel(int *input, int *output, int n, int block_size)
{
   extern __shared__ int temp[];

   int tid = threadIdx.x;
   int block_idx = blockIdx.x;
   int block_offset = block_idx * block_size;

   // Load last element of each block
   if (block_offset + block_size - 1 < n)
   {
      temp[tid] = input[block_offset + block_size - 1];
   }
   else if (block_offset < n)
   {
      temp[tid] = input[n - 1];
   }
   else
   {
      temp[tid] = 0;
   }

   __syncthreads();

   // Perform a simple parallel reduction
   for (int stride = 1; stride < blockDim.x; stride *= 2)
   {
      int index = 2 * stride * (tid + 1) - 1;
      if (index < blockDim.x)
      {
         temp[index] += temp[index - stride];
      }
      __syncthreads();
   }

   // Write the block sum
   if (tid == 0 && block_idx < gridDim.x)
   {
      output[block_idx] = temp[blockDim.x - 1];
   }
}

__global__ void add_block_sums_kernel(int *input, int *block_sums, int n, int block_size)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int block_idx = blockIdx.x;

   if (idx < n && block_idx > 0)
   {
      input[idx] += block_sums[block_idx - 1];
   }
}

// Host function to execute the naive scan
void naive_scan(int *d_input, int *d_output, int n, float *elapsed_time)
{
   int block_size = 512; // Typical block size
   int grid_size = (n + block_size - 1) / block_size;

   // Create CUDA events for timing
   cudaEvent_t start, stop;
   checkCudaErrors(cudaEventCreate(&start));
   checkCudaErrors(cudaEventCreate(&stop));

   // Record start event
   checkCudaErrors(cudaEventRecord(start));

   // Launch the naive scan kernel
   naive_scan_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, n);
   checkCudaErrors(cudaGetLastError());

   // Record stop event and synchronize
   checkCudaErrors(cudaEventRecord(stop));
   checkCudaErrors(cudaEventSynchronize(stop));

   // Calculate elapsed time
   checkCudaErrors(cudaEventElapsedTime(elapsed_time, start, stop));

   // Clean up events
   checkCudaErrors(cudaEventDestroy(start));
   checkCudaErrors(cudaEventDestroy(stop));
}

// Host function to execute the optimized scan
void optimized_scan(int *d_input, int *d_output, int n, float *elapsed_time)
{
   int block_size = 512; // Typical block size
   int grid_size = (n + block_size - 1) / block_size;

   // For large arrays, we need a multi-block approach
   int *d_block_sums = nullptr;
   if (grid_size > 1)
   {
      checkCudaErrors(cudaMalloc(&d_block_sums, grid_size * sizeof(int)));
   }

   // Create CUDA events for timing
   cudaEvent_t start, stop;
   checkCudaErrors(cudaEventCreate(&start));
   checkCudaErrors(cudaEventCreate(&stop));

   // Record start event
   checkCudaErrors(cudaEventRecord(start));

   if (grid_size == 1)
   {
      // Simple case: one block is enough
      work_efficient_scan_kernel<<<1, block_size, block_size * sizeof(int)>>>(d_input, d_output, n);
      checkCudaErrors(cudaGetLastError());
   }
   else
   {
      // Multi-block approach
      // Step 1: Perform scan within each block
      work_efficient_scan_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, n);
      checkCudaErrors(cudaGetLastError());

      // Step 2: Compute the sum of each block
      block_sum_kernel<<<1, grid_size, grid_size * sizeof(int)>>>(d_output, d_block_sums, n, block_size);
      checkCudaErrors(cudaGetLastError());

      // Step 3: Scan the block sums
      work_efficient_scan_kernel<<<1, grid_size, grid_size * sizeof(int)>>>(d_block_sums, d_block_sums, grid_size);
      checkCudaErrors(cudaGetLastError());

      // Step 4: Add the scanned block sums back
      add_block_sums_kernel<<<grid_size, block_size>>>(d_output, d_block_sums, n, block_size);
      checkCudaErrors(cudaGetLastError());
   }

   // Record stop event and synchronize
   checkCudaErrors(cudaEventRecord(stop));
   checkCudaErrors(cudaEventSynchronize(stop));

   // Calculate elapsed time
   checkCudaErrors(cudaEventElapsedTime(elapsed_time, start, stop));

   // Clean up events and temporary memory
   checkCudaErrors(cudaEventDestroy(start));
   checkCudaErrors(cudaEventDestroy(stop));

   if (grid_size > 1)
   {
      checkCudaErrors(cudaFree(d_block_sums));
   }
}

int main(int argc, char **argv)
{
   // Parse command line arguments
   int n = (argc > 1) ? atoi(argv[1]) : 1024; // Default size if not specified

   printf("Running prefix sum on array of size %d\n", n);

   // Allocate host memory
   int *h_input = (int *)malloc(n * sizeof(int));
   int *h_output_seq = (int *)malloc(n * sizeof(int));
   int *h_output_naive = (int *)malloc(n * sizeof(int));
   int *h_output_opt = (int *)malloc(n * sizeof(int));

   // Initialize input array with random values
   srand(42); // For reproducibility
   for (int i = 0; i < n; i++)
   {
      h_input[i] = rand() % 10; // Random values between 0 and 9
   }

   // Print first 10 elements of input array
   printf("\nInput array (first 10 elements):\n");
   for (int i = 0; i < 10 && i < n; i++)
   {
      printf("%d ", h_input[i]);
   }
   printf("\n");

   // Allocate device memory
   int *d_input;
   int *d_output_naive;
   int *d_output_opt;
   checkCudaErrors(cudaMalloc(&d_input, n * sizeof(int)));
   checkCudaErrors(cudaMalloc(&d_output_naive, n * sizeof(int)));
   checkCudaErrors(cudaMalloc(&d_output_opt, n * sizeof(int)));

   // Copy input data to device
   checkCudaErrors(cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice));

   // Execute sequential scan and measure time
   float seq_time = 0.0f;
   cudaEvent_t seq_start, seq_stop;
   checkCudaErrors(cudaEventCreate(&seq_start));
   checkCudaErrors(cudaEventCreate(&seq_stop));

   checkCudaErrors(cudaEventRecord(seq_start));
   sequential_scan(h_input, h_output_seq, n);
   checkCudaErrors(cudaEventRecord(seq_stop));
   checkCudaErrors(cudaEventSynchronize(seq_stop));
   checkCudaErrors(cudaEventElapsedTime(&seq_time, seq_start, seq_stop));

   // Execute naive parallel scan
   float naive_time = 0.0f;
   naive_scan(d_input, d_output_naive, n, &naive_time);

   // Execute optimized parallel scan
   float opt_time = 0.0f;
   optimized_scan(d_input, d_output_opt, n, &opt_time);

   // Copy results back to host
   checkCudaErrors(cudaMemcpy(h_output_naive, d_output_naive, n * sizeof(int), cudaMemcpyDeviceToHost));
   checkCudaErrors(cudaMemcpy(h_output_opt, d_output_opt, n * sizeof(int), cudaMemcpyDeviceToHost));

   // Print first 10 elements of output arrays
   printf("\nSequential scan result (first 10 elements):\n");
   for (int i = 0; i < 10 && i < n; i++)
   {
      printf("%d ", h_output_seq[i]);
   }
   printf("\n");

   printf("\nNaive parallel scan result (first 10 elements):\n");
   for (int i = 0; i < 10 && i < n; i++)
   {
      printf("%d ", h_output_naive[i]);
   }
   printf("\n");

   printf("\nOptimized parallel scan result (first 10 elements):\n");
   for (int i = 0; i < 10 && i < n; i++)
   {
      printf("%d ", h_output_opt[i]);
   }
   printf("\n");

   // Print timing results and speedups
   printf("\nExecution times:\n");
   printf("Sequential: %.3f ms\n", seq_time);
   printf("Naive parallel: %.3f ms\n", naive_time);
   printf("Optimized parallel: %.3f ms\n", opt_time);

   printf("\nSpeedups:\n");
   printf("Naive vs Sequential: %.2fx\n", seq_time / naive_time);
   printf("Optimized vs Sequential: %.2fx\n", seq_time / opt_time);
   printf("Optimized vs Naive: %.2fx\n", naive_time / opt_time);

   // Verify results
   bool naive_correct = true;
   bool opt_correct = true;
   for (int i = 0; i < n; i++)
   {
      if (h_output_seq[i] != h_output_naive[i])
      {
         naive_correct = false;
         break;
      }
      if (h_output_seq[i] != h_output_opt[i])
      {
         opt_correct = false;
         break;
      }
   }

   printf("\nResult verification:\n");
   printf("Naive implementation: %s\n", naive_correct ? "CORRECT" : "INCORRECT");
   printf("Optimized implementation: %s\n", opt_correct ? "CORRECT" : "INCORRECT");

   // Clean up
   free(h_input);
   free(h_output_seq);
   free(h_output_naive);
   free(h_output_opt);

   checkCudaErrors(cudaFree(d_input));
   checkCudaErrors(cudaFree(d_output_naive));
   checkCudaErrors(cudaFree(d_output_opt));

   checkCudaErrors(cudaEventDestroy(seq_start));
   checkCudaErrors(cudaEventDestroy(seq_stop));

   return 0;
}