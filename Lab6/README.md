# HPC Lab 6: Parallel Array and Matrix Operations (Python)

## Description

This project implements parallel computing solutions using Python's multiprocessing and NumPy for high-performance array and matrix operations. The code demonstrates various parallel programming concepts including process pools, load balancing, performance analysis, and scalability testing.

## Project Structure

```
lab6/
├── parallel_array.py        # Main parallel array operations
├── matrix_operations.py     # Matrix multiplication and addition
├── performance_analysis.py  # Performance benchmarking tools
├── Makefile                # Build and execution configuration
├── requirements.txt        # Python dependencies (auto-generated)
├── README.md               # This file
└── HPC-Lab6.pdf           # Lab assignment document
```

## Code Sources Description

### 1. parallel_array.py
This file contains the main implementation of parallel array operations using Python:

- **ParallelArrayProcessor Class**: Manages array data and parallel operations
- **Key Features**:
  - Array initialization with NumPy random values
  - Parallel sum calculation using ProcessPoolExecutor
  - Sequential vs parallel comparison
  - Parallel maximum finding with chunked processing
  - Vector multiplication operations using NumPy
  - Parallel transformations (e.g., sqrt operations)
  - Performance timing for all operations

- **Operations Implemented**:
  - `parallel_sum_chunks()`: Computes array sum using multiprocessing
  - `sequential_sum()`: Sequential baseline using NumPy
  - `find_max_parallel()`: Finds maximum value using parallel chunks
  - `vector_multiplication()`: Element-wise vector multiplication
  - `parallel_transform()`: Apply functions in parallel across chunks

### 2. matrix_operations.py
Implements parallel matrix operations for performance comparison:

- **MatrixOperations Class**: Handles matrix data and computations
- **Key Features**:
  - Square matrix initialization with NumPy random values
  - Multiple parallel strategies (row-wise, block-wise)
  - NumPy optimized operations for baseline
  - Result verification between different methods
  - Performance timing and result display

- **Operations Implemented**:
  - `sequential_multiplication()`: Traditional O(n³) matrix multiplication
  - `numpy_multiplication()`: NumPy's optimized BLAS routines
  - `parallel_multiplication_rows()`: Row-wise parallel distribution
  - `parallel_multiplication_blocks()`: Block decomposition approach
  - `parallel_addition()`: Element-wise matrix addition

### 3. performance_analysis.py
Comprehensive performance benchmarking and system analysis tools:

- **PerformanceBenchmark Class**: Systematic performance testing
- **LoadBalancingDemo Class**: Demonstrates scheduling strategies
- **SystemProfiler Class**: System resource monitoring
- **Key Features**:
  - Multi-process performance testing (1, 2, 4, 8, 16+ workers)
  - Scalability analysis with speedup and efficiency calculations
  - Load balancing comparison (static vs dynamic task distribution)
  - JSON results logging for analysis
  - System resource monitoring (CPU, memory usage)
  - Uneven workload simulation

- **Analysis Metrics**:
  - Execution time measurement with microsecond precision
  - Speedup calculation (sequential_time / parallel_time)
  - Parallel efficiency (speedup / workers × 100%)
  - Resource utilization monitoring

## Dependencies

- **Python**: 3.6 or higher
- **NumPy**: For efficient array operations and linear algebra
- **Matplotlib**: For plotting and visualization (optional)
- **Pandas**: For data analysis and results processing
- **psutil**: For system resource monitoring
- **Standard Libraries**: 
  - `multiprocessing` for parallel processing
  - `concurrent.futures` for thread/process pools
  - `time` for high-precision timing
  - `json` for results serialization

