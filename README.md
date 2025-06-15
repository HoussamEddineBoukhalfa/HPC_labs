# ⚙️ High Performance Computing Labs

This repository contains all completed practical labs for the **High Performance Computing (HPC)** module at ENSIA (2025). Each lab explores a key paradigm in parallel and distributed programming using real-world tools such as MPI, OpenMP, CUDA, and Ray.

## 📚 Lab Overview

### 🔁 Lab 1 – MPI Fundamentals (C & Python)
> **Tools**: OpenMPI, mpi4py  
- Hello MPI World (rank/size detection)
- Point-to-point communication with `Send`/`Recv`
- Ring communication pattern

### 🔗 Lab 2 – MPI for Machine Learning
> **Tools**: mpi4py, NumPy, Scikit-learn  
- Parallel data broadcasting
- Parallel K-means clustering
- Gradient averaging in linear regression
- Distributed model evaluation

### 🧵 Lab 3 – OpenMP Parallelism
> **Tools**: OpenMP (C)  
- Basic parallel regions
- Vector addition and matrix multiplication
- Shared resource synchronization with atomic constructs

### 🔬 Lab 4 – CUDA Programming Basics
> **Tools**: CUDA C/C++  
- Vector addition with kernel launches
- ReLU activation and matrix-vector multiplication
- Shared memory reduction (dot product)
- 1D stencil, Euclidean distance, partial argmax, 2D matrix addition

### ⚡ Lab 5 – Parallel Prefix Sum in CUDA
> **Tools**: CUDA C/C++  
- Naive vs optimized scan algorithms
- Shared memory efficiency
- Performance benchmarking with CUDA events

### ☁️ Lab 6 – Distributed Computing with Ray
> **Tools**: Ray, Python, NumPy, Scikit-learn  
- Cluster setup across CPU/GPU nodes
- Remote task execution
- GPU usage with `@ray.remote(num_gpus=...)`
- Distributed ML with Ray and K-Fold cross-validation

---

## 📂 Repository Structure

HPC_labs/
├── Lab1/ # MPI basics (C & Python)
├── Lab2/ # MPI for ML (Python)
├── Lab3/ # OpenMP in C
├── Lab4/ # Intro to CUDA C/C++
├── Lab5/ # Parallel Scan with CUDA
├── Lab6/ # Distributed Ray with Python
└── README.md

---

## 🚀 How to Run

### MPI (C)
```bash
mpicc file.c -o file
mpiexec -n 4 ./file

MPI (Python)

mpiexec -n 4 python file.py

OpenMP

gcc -fopenmp file.c -o file
export OMP_NUM_THREADS=4
./file

CUDA

nvcc file.cu -o file -lm
./file

Ray

# Setup virtual environment and install Ray
pip install "ray[default]"
python script.py

👨‍🎓 Author

Houssam Eddine Boukhalfa
Fourth-Year AI Engineering Student
National School of Artificial Intelligence – ENSIA