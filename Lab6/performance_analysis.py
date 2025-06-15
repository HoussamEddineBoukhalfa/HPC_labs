#!/usr/bin/env python3
"""
HPC Lab 6: Performance Analysis in Python
Comprehensive benchmarking and scalability analysis for parallel operations
"""

import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd
import json
from functools import partial
import psutil
import sys
import os

class PerformanceBenchmark:
    def __init__(self, size, max_workers=None):
        """
        Initialize performance benchmark
        
        Args:
            size (int): Size of data to process
            max_workers (int): Maximum number of workers to test
        """
        self.size = size
        self.max_workers = max_workers or mp.cpu_count()
        self.data = None
        self.results = {}
        self.initialize_data()
        
    def initialize_data(self):
        """Initialize data with random values"""
        np.random.seed(42)
        self.data = np.random.uniform(1.0, 1000.0, self.size)
        
    def measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000000, result  # Return microseconds
    
    def sequential_operations(self):
        """Perform sequential operations for baseline"""
        operations = {}
        
        # Sequential sum
        exec_time, result = self.measure_time(np.sum, self.data)
        operations['sum'] = {'time': exec_time, 'result': result}
        
        # Sequential max
        exec_time, result = self.measure_time(np.max, self.data)
        operations['max'] = {'time': exec_time, 'result': result}
        
        # Sequential transform (sqrt * 2.5)
        exec_time, result = self.measure_time(lambda x: np.sqrt(x) * 2.5, self.data)
        operations['transform'] = {'time': exec_time, 'result': result}
        
        # Sequential mean
        exec_time, result = self.measure_time(np.mean, self.data)
        operations['mean'] = {'time': exec_time, 'result': result}
        
        return operations
    
    def parallel_operations(self, num_workers):
        """Perform parallel operations with specified number of workers"""
        operations = {}
        
        # Split data into chunks
        chunk_size = max(1, self.size // num_workers)
        chunks = [self.data[i:i + chunk_size] for i in range(0, self.size, chunk_size)]
        
        # Parallel sum
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            partial_sums = list(executor.map(np.sum, chunks))
        result = sum(partial_sums)
        end_time = time.perf_counter()
        operations['sum'] = {'time': (end_time - start_time) * 1000000, 'result': result}
        
        # Parallel max
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            partial_maxes = list(executor.map(np.max, chunks))
        result = max(partial_maxes)
        end_time = time.perf_counter()
        operations['max'] = {'time': (end_time - start_time) * 1000000, 'result': result}
        
        # Parallel transform
        def transform_chunk(chunk):
            return np.sqrt(chunk) * 2.5
        
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(transform_chunk, chunks))
        result = np.concatenate(results)
        end_time = time.perf_counter()
        operations['transform'] = {'time': (end_time - start_time) * 1000000, 'result': result}
        
        # Parallel mean
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            partial_sums = list(executor.map(np.sum, chunks))
            partial_counts = list(executor.map(len, chunks))
        result = sum(partial_sums) / sum(partial_counts)
        end_time = time.perf_counter()
        operations['mean'] = {'time': (end_time - start_time) * 1000000, 'result': result}
        
        return operations
    
    def run_scalability_test(self):
        """Run scalability test with different numbers of workers"""
        print(f"\n=== Scalability Test for size {self.size} ===")
        
        # Get sequential baseline
        sequential_results = self.sequential_operations()
        self.results['sequential'] = sequential_results
        
        print(f"Sequential baseline:")
        for op, data in sequential_results.items():
            print(f"  {op}: {data['time']:.0f} µs")
        
        # Test different worker counts
        worker_counts = [1, 2, 4, 8, 16, 32]
        worker_counts = [w for w in worker_counts if w <= self.max_workers]
        
        for workers in worker_counts:
            print(f"\nTesting with {workers} workers:")
            parallel_results = self.parallel_operations(workers)
            self.results[f'parallel_{workers}'] = parallel_results
            
            for op, data in parallel_results.items():
                seq_time = sequential_results[op]['time']
                par_time = data['time']
                speedup = seq_time / par_time if par_time > 0 else 0
                efficiency = (speedup / workers) * 100 if workers > 0 else 0
                
                print(f"  {op}: {par_time:.0f} µs, speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%")
    
    def save_results(self, filename):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, operations in self.results.items():
                json_results[key] = {}
                for op, data in operations.items():
                    json_results[key][op] = {
                        'time': data['time'],
                        'result': float(data['result']) if np.isscalar(data['result']) else data['result'].tolist()
                    }
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {filename}")

class LoadBalancingDemo:
    """Demonstrate load balancing with uneven workloads"""
    
    @staticmethod
    def create_uneven_workload(size, heavy_work_ratio=0.1):
        """Create workload where some tasks take much longer"""
        workload = np.ones(size, dtype=int)
        heavy_indices = np.random.choice(size, int(size * heavy_work_ratio), replace=False)
        workload[heavy_indices] = 1000  # Heavy tasks
        return workload
    
    @staticmethod
    def simulate_work(work_amount):
        """Simulate computational work"""
        result = 0
        for i in range(work_amount):
            result += i * i
        return result
    
    @staticmethod
    def process_chunk_static(chunk):
        """Process a chunk of work (static assignment)"""
        return [LoadBalancingDemo.simulate_work(work) for work in chunk]
    
    @staticmethod
    def process_single_task(work):
        """Process a single task (for dynamic assignment)"""
        return LoadBalancingDemo.simulate_work(work)
    
    @classmethod
    def compare_scheduling(cls, size=10000):
        """Compare static vs dynamic load balancing"""
        print(f"\n=== Load Balancing Demonstration ===")
        print(f"Workload size: {size}")
        
        workload = cls.create_uneven_workload(size)
        num_workers = mp.cpu_count()
        
        print(f"Heavy tasks: {np.sum(workload > 1)} out of {size}")
        print(f"Workers: {num_workers}")
        
        # Static scheduling - divide into equal chunks
        print(f"\n1. Static Scheduling:")
        chunk_size = size // num_workers
        chunks = [workload[i:i + chunk_size] for i in range(0, size, chunk_size)]
        
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(cls.process_chunk_static, chunks))
        static_time = time.perf_counter() - start_time
        
        print(f"Static scheduling time: {static_time:.3f} seconds")
        
        # Dynamic scheduling - submit individual tasks
        print(f"\n2. Dynamic Scheduling:")
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(cls.process_single_task, workload))
        dynamic_time = time.perf_counter() - start_time
        
        print(f"Dynamic scheduling time: {dynamic_time:.3f} seconds")
        
        # Analysis
        improvement = static_time / dynamic_time
        print(f"\nDynamic is {improvement:.2f}x faster than static")
        
        return static_time, dynamic_time

class SystemProfiler:
    """Profile system resources during execution"""
    
    @staticmethod
    def get_system_info():
        """Get system information"""
        info = {
            'cpu_count': mp.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory': psutil.virtual_memory()._asdict(),
            'python_version': sys.version,
            'numpy_version': np.__version__
        }
        return info
    
    @staticmethod
    def monitor_resources(duration=1.0):
        """Monitor CPU and memory usage"""
        start_time = time.time()
        cpu_usage = []
        memory_usage = []
        
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=0.1))
            memory_usage.append(psutil.virtual_memory().percent)
        
        return {
            'avg_cpu': np.mean(cpu_usage),
            'max_cpu': np.max(cpu_usage),
            'avg_memory': np.mean(memory_usage),
            'max_memory': np.max(memory_usage)
        }

def create_performance_report(results_file="performance_results.json"):
    """Create a comprehensive performance report"""
    print(f"\n=== Performance Report ===")
    
    # System information
    system_info = SystemProfiler.get_system_info()
    print(f"System Info:")
    print(f"  CPU cores: {system_info['cpu_count']}")
    print(f"  Memory: {system_info['memory']['total'] / (1024**3):.1f} GB")
    print(f"  Python: {system_info['python_version'].split()[0]}")
    print(f"  NumPy: {system_info['numpy_version']}")

def main():
    print("=== HPC Lab 6: Performance Analysis in Python ===")
    
    # System information
    system_info = SystemProfiler.get_system_info()
    print(f"System: {system_info['cpu_count']} cores, "
          f"{system_info['memory']['total'] / (1024**3):.1f} GB RAM")
    
    # Test different data sizes
    sizes = [10000, 100000, 1000000, 5000000]
    
    for size in sizes:
        print(f"\n{'='*60}")
        benchmark = PerformanceBenchmark(size)
        benchmark.run_scalability_test()
        
        # Save results
        results_filename = f"performance_results_{size}.json"
        benchmark.save_results(results_filename)
    
    # Load balancing demonstration
    LoadBalancingDemo.compare_scheduling(10000)
    
    # Resource monitoring example
    print(f"\n=== Resource Monitoring Example ===")
    print("Monitoring system resources during computation...")
    
    start_time = time.time()
    # Simulate some work
    data = np.random.random(1000000)
    result = np.sum(data ** 2)
    
    resources = SystemProfiler.monitor_resources(1.0)
    print(f"Resource usage:")
    print(f"  Average CPU: {resources['avg_cpu']:.1f}%")
    print(f"  Peak CPU: {resources['max_cpu']:.1f}%")
    print(f"  Average Memory: {resources['avg_memory']:.1f}%")
    
    print(f"\nPerformance analysis complete!")
    print(f"Results saved to performance_results_*.json files")

if __name__ == "__main__":
    # Ensure multiprocessing works properly
    mp.set_start_method('spawn', force=True)
    main()