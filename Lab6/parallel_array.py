#!/usr/bin/env python3
"""
HPC Lab 6: Parallel Array Operations in Python
Demonstrates parallel computing using multiprocessing and concurrent.futures
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import random
from functools import partial
import sys

class ParallelArrayProcessor:
    def __init__(self, size, num_processes=None):
        """
        Initialize the parallel array processor
        
        Args:
            size (int): Size of the array
            num_processes (int): Number of processes to use (default: CPU count)
        """
        self.size = size
        self.num_processes = num_processes or mp.cpu_count()
        self.data = None
        self.initialize_array()
        
    def initialize_array(self):
        """Initialize array with random values"""
        np.random.seed(42)  # For reproducible results
        self.data = np.random.uniform(1.0, 100.0, self.size)
        
    def sequential_sum(self):
        """Compute sum sequentially"""
        start_time = time.perf_counter()
        result = np.sum(self.data)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000000  # Convert to microseconds
        print(f"Sequential sum time: {execution_time:.0f} microseconds")
        return result
        
    def parallel_sum_chunks(self):
        """Compute sum using parallel processing with chunks"""
        start_time = time.perf_counter()
        
        # Split array into chunks
        chunk_size = self.size // self.num_processes
        chunks = [self.data[i:i + chunk_size] for i in range(0, self.size, chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            partial_sums = list(executor.map(np.sum, chunks))
        
        result = sum(partial_sums)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000000
        print(f"Parallel sum time: {execution_time:.0f} microseconds")
        return result
        
    def parallel_sum_numpy(self):
        """Compute sum using NumPy's optimized parallel operations"""
        start_time = time.perf_counter()
        result = np.sum(self.data)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000000
        print(f"NumPy optimized sum time: {execution_time:.0f} microseconds")
        return result
        
    def find_max_parallel(self):
        """Find maximum value using parallel processing"""
        start_time = time.perf_counter()
        
        chunk_size = self.size // self.num_processes
        chunks = [self.data[i:i + chunk_size] for i in range(0, self.size, chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            partial_maxes = list(executor.map(np.max, chunks))
        
        result = max(partial_maxes)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000000
        print(f"Parallel max search time: {execution_time:.0f} microseconds")
        return result
        
    def vector_multiplication(self, other_vector):
        """Element-wise vector multiplication"""
        if len(other_vector) != self.size:
            raise ValueError("Vector size mismatch!")
            
        start_time = time.perf_counter()
        result = self.data * other_vector
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000000
        print(f"Vector multiplication time: {execution_time:.0f} microseconds")
        return result
        
    def parallel_transform(self, transform_func):
        """Apply transformation function in parallel"""
        start_time = time.perf_counter()
        
        chunk_size = self.size // self.num_processes
        chunks = [self.data[i:i + chunk_size] for i in range(0, self.size, chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(transform_func, chunks))
        
        # Combine results
        result = np.concatenate(results)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000000
        print(f"Parallel transform time: {execution_time:.0f} microseconds")
        return result
        
    def print_stats(self):
        """Print array statistics"""
        print(f"\n=== Array Statistics ===")
        print(f"Array size: {self.size}")
        print(f"Number of processes: {self.num_processes}")
        print(f"First 10 elements: {self.data[:10]}")
        print(f"Mean: {np.mean(self.data):.2f}")
        print(f"Std Dev: {np.std(self.data):.2f}")

def sqrt_transform(arr):
    """Square root transformation function"""
    return np.sqrt(arr) * 2.5

def main():
    print("=== HPC Lab 6: Parallel Array Operations in Python ===")
    
    # Test different array sizes
    sizes = [1000, 10000, 100000, 1000000]
    num_processes = mp.cpu_count()
    
    print(f"Available CPU cores: {num_processes}")
    print(f"NumPy using {np.show_config(mode='dicts')}")
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"--- Testing with array size: {size} ---")
        
        processor = ParallelArrayProcessor(size, num_processes)
        processor.print_stats()
        
        # Test sum operations
        print(f"\n1. Sum Operations:")
        seq_sum = processor.sequential_sum()
        par_sum = processor.parallel_sum_chunks()
        numpy_sum = processor.parallel_sum_numpy()
        
        print(f"Sequential sum: {seq_sum:.2f}")
        print(f"Parallel sum: {par_sum:.2f}")
        print(f"NumPy sum: {numpy_sum:.2f}")
        print(f"Difference (seq vs par): {abs(seq_sum - par_sum):.6f}")
        
        # Test max finding
        print(f"\n2. Maximum Finding:")
        max_val = processor.find_max_parallel()
        print(f"Maximum value: {max_val:.2f}")
        
        # Test vector multiplication
        print(f"\n3. Vector Multiplication:")
        other_vector = np.full(size, 2.0)
        result = processor.vector_multiplication(other_vector)
        print(f"First 5 multiplication results: {result[:5]}")
        
        # Test parallel transformation
        print(f"\n4. Parallel Transformation (sqrt * 2.5):")
        transformed = processor.parallel_transform(sqrt_transform)
        print(f"First 5 transformed values: {transformed[:5]}")
        
        print(f"{'='*50}")

if __name__ == "__main__":
    main()