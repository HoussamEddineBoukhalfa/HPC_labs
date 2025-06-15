#!/usr/bin/env python3
"""
HPC Lab 6: Matrix Operations in Python
Demonstrates parallel matrix operations using NumPy and multiprocessing
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import sys

class MatrixOperations:
    def __init__(self, size, num_processes=None):
        """
        Initialize matrix operations
        
        Args:
            size (int): Size of square matrices (size x size)
            num_processes (int): Number of processes to use
        """
        self.size = size
        self.num_processes = num_processes or mp.cpu_count()
        self.matrix_a = None
        self.matrix_b = None
        self.result = None
        self.initialize_matrices()
        
    def initialize_matrices(self):
        """Initialize matrices with random values"""
        np.random.seed(42)  # For reproducible results
        self.matrix_a = np.random.uniform(0.0, 10.0, (self.size, self.size))
        self.matrix_b = np.random.uniform(0.0, 10.0, (self.size, self.size))
        self.result = np.zeros((self.size, self.size))
        
    def sequential_multiplication(self):
        """Sequential matrix multiplication using nested loops"""
        start_time = time.perf_counter()
        
        result = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    result[i, j] += self.matrix_a[i, k] * self.matrix_b[k, j]
        
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Sequential matrix multiplication: {execution_time:.2f} ms")
        return result
        
    def numpy_multiplication(self):
        """Matrix multiplication using NumPy's optimized routines"""
        start_time = time.perf_counter()
        result = np.dot(self.matrix_a, self.matrix_b)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000
        print(f"NumPy optimized multiplication: {execution_time:.2f} ms")
        return result
        
    def parallel_multiplication_rows(self):
        """Parallel matrix multiplication by distributing rows"""
        start_time = time.perf_counter()
        
        def multiply_row_chunk(row_indices):
            """Multiply specific rows of matrix A with matrix B"""
            result_chunk = np.zeros((len(row_indices), self.size))
            for idx, i in enumerate(row_indices):
                for j in range(self.size):
                    for k in range(self.size):
                        result_chunk[idx, j] += self.matrix_a[i, k] * self.matrix_b[k, j]
            return result_chunk
        
        # Split rows among processes
        rows_per_process = self.size // self.num_processes
        row_chunks = []
        for i in range(self.num_processes):
            start_row = i * rows_per_process
            end_row = (i + 1) * rows_per_process if i < self.num_processes - 1 else self.size
            row_chunks.append(list(range(start_row, end_row)))
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(multiply_row_chunk, row_chunks))
        
        # Combine results
        result = np.vstack(results)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000
        print(f"Parallel matrix multiplication (rows): {execution_time:.2f} ms")
        return result
        
    def parallel_multiplication_blocks(self):
        """Parallel matrix multiplication using block decomposition"""
        start_time = time.perf_counter()
        
        def multiply_block(args):
            """Multiply a block of the result matrix"""
            i_start, i_end, j_start, j_end = args
            block_result = np.zeros((i_end - i_start, j_end - j_start))
            
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(self.size):
                        block_result[i - i_start, j - j_start] += \
                            self.matrix_a[i, k] * self.matrix_b[k, j]
            
            return (i_start, i_end, j_start, j_end, block_result)
        
        # Create block assignments
        block_size = max(1, self.size // int(np.sqrt(self.num_processes)))
        blocks = []
        for i in range(0, self.size, block_size):
            for j in range(0, self.size, block_size):
                i_end = min(i + block_size, self.size)
                j_end = min(j + block_size, self.size)
                blocks.append((i, i_end, j, j_end))
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            block_results = list(executor.map(multiply_block, blocks))
        
        # Combine block results
        result = np.zeros((self.size, self.size))
        for i_start, i_end, j_start, j_end, block_data in block_results:
            result[i_start:i_end, j_start:j_end] = block_data
        
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        print(f"Parallel matrix multiplication (blocks): {execution_time:.2f} ms")
        return result
        
    def parallel_addition(self):
        """Parallel matrix addition"""
        start_time = time.perf_counter()
        
        def add_chunk(args):
            """Add a chunk of matrices"""
            start_idx, end_idx = args
            return self.matrix_a.flat[start_idx:end_idx] + self.matrix_b.flat[start_idx:end_idx]
        
        # Split flattened matrices into chunks
        total_elements = self.size * self.size
        chunk_size = total_elements // self.num_processes
        chunks = []
        for i in range(self.num_processes):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.num_processes - 1 else total_elements
            chunks.append((start_idx, end_idx))
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(add_chunk, chunks))
        
        # Combine results and reshape
        result_flat = np.concatenate(results)
        result = result_flat.reshape((self.size, self.size))
        
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        print(f"Parallel matrix addition: {execution_time:.2f} ms")
        return result
        
    def numpy_addition(self):
        """Matrix addition using NumPy"""
        start_time = time.perf_counter()
        result = self.matrix_a + self.matrix_b
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000
        print(f"NumPy matrix addition: {execution_time:.2f} ms")
        return result
        
    def print_matrix(self, matrix, name, max_size=5):
        """Print a portion of the matrix"""
        print(f"\n{name} (first {max_size}x{max_size}):")
        display_size = min(max_size, self.size)
        for i in range(display_size):
            row_str = "\t".join([f"{matrix[i, j]:.2f}" for j in range(display_size)])
            if self.size > max_size:
                row_str += "\t..."
            print(row_str)
        if self.size > max_size:
            print("\t...")
            
    def verify_results(self, result1, result2, tolerance=1e-10):
        """Verify that two matrices are approximately equal"""
        if np.allclose(result1, result2, atol=tolerance):
            print("✓ Results match!")
            return True
        else:
            max_diff = np.max(np.abs(result1 - result2))
            print(f"✗ Results differ! Maximum difference: {max_diff}")
            return False
            
    def print_stats(self):
        """Print matrix statistics"""
        print(f"\n=== Matrix Statistics ===")
        print(f"Matrix size: {self.size}x{self.size}")
        print(f"Number of processes: {self.num_processes}")
        print(f"Matrix A - Mean: {np.mean(self.matrix_a):.2f}, Std: {np.std(self.matrix_a):.2f}")
        print(f"Matrix B - Mean: {np.mean(self.matrix_b):.2f}, Std: {np.std(self.matrix_b):.2f}")

def main():
    print("=== HPC Lab 6: Matrix Operations in Python ===")
    
    sizes = [50, 100, 200, 500]  # Smaller sizes for demonstration
    num_processes = mp.cpu_count()
    
    print(f"Available CPU cores: {num_processes}")
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"--- Matrix size: {size}x{size} ---")
        
        matrices = MatrixOperations(size, num_processes)
        matrices.print_stats()
        
        print(f"\nPerforming matrix operations...")
        
        # Matrix addition comparison
        print(f"\n1. Matrix Addition:")
        numpy_add_result = matrices.numpy_addition()
        parallel_add_result = matrices.parallel_addition()
        matrices.verify_results(numpy_add_result, parallel_add_result)
        
        # Matrix multiplication comparison
        print(f"\n2. Matrix Multiplication:")
        numpy_mult_result = matrices.numpy_multiplication()
        
        if size <= 200:  # Only run slower methods for smaller matrices
            sequential_mult_result = matrices.sequential_multiplication()
            parallel_rows_result = matrices.parallel_multiplication_rows()
            parallel_blocks_result = matrices.parallel_multiplication_blocks()
            
            print(f"\nVerifying multiplication results:")
            matrices.verify_results(numpy_mult_result, sequential_mult_result)
            matrices.verify_results(numpy_mult_result, parallel_rows_result)
            matrices.verify_results(numpy_mult_result, parallel_blocks_result)
        
        # Display sample results for small matrices
        if size <= 100:
            matrices.print_matrix(matrices.matrix_a, "Matrix A")
            matrices.print_matrix(matrices.matrix_b, "Matrix B")
            matrices.print_matrix(numpy_mult_result, "Result Matrix")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    main()