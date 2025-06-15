# Python-based HPC Lab 6 Makefile
PYTHON = python3
VENV_DIR = venv
REQUIREMENTS = requirements.txt

# Python source files
SOURCES = parallel_array.py matrix_operations.py performance_analysis.py

# Default target
all: check-python install-deps

# Check Python installation
check-python:
	@echo "Checking Python installation..."
	@$(PYTHON) --version || (echo "Python 3 not found! Please install Python 3." && exit 1)
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3,6) else 1)" || (echo "Python 3.6+ required!" && exit 1)
	@echo "✓ Python installation OK"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "✓ Virtual environment created"

# Install dependencies
install-deps: $(REQUIREMENTS)
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)
	@echo "✓ Dependencies installed"

# Create requirements.txt if it doesn't exist
$(REQUIREMENTS):
	@echo "Creating requirements.txt..."
	@echo "numpy>=1.20.0" > $(REQUIREMENTS)
	@echo "matplotlib>=3.3.0" >> $(REQUIREMENTS)
	@echo "pandas>=1.3.0" >> $(REQUIREMENTS)
	@echo "psutil>=5.8.0" >> $(REQUIREMENTS)
	@echo "✓ Requirements file created"

# Run all programs
run: all
	@echo "Running Parallel Array Operations..."
	$(PYTHON) parallel_array.py
	@echo -e "\n\nRunning Matrix Operations..."
	$(PYTHON) matrix_operations.py
	@echo -e "\n\nRunning Performance Analysis..."
	$(PYTHON) performance_analysis.py

# Run individual programs
run-array: check-python
	$(PYTHON) parallel_array.py

run-matrix: check-python
	$(PYTHON) matrix_operations.py

run-performance: check-python
	$(PYTHON) performance_analysis.py

# Test installations
test-deps:
	@echo "Testing dependencies..."
	@$(PYTHON) -c "import numpy; print(f'NumPy {numpy.__version__} ✓')" || echo "NumPy not installed"
	@$(PYTHON) -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__} ✓')" || echo "Matplotlib not installed"
	@$(PYTHON) -c "import pandas; print(f'Pandas {pandas.__version__} ✓')" || echo "Pandas not installed"
	@$(PYTHON) -c "import psutil; print(f'psutil {psutil.__version__} ✓')" || echo "psutil not installed"
	@$(PYTHON) -c "import multiprocessing; print(f'Multiprocessing available: {multiprocessing.cpu_count()} cores ✓')"

# Check system capabilities
check-system:
	@echo "System Information:"
	@$(PYTHON) -c "import multiprocessing, psutil; print(f'CPU cores: {multiprocessing.cpu_count()}'); print(f'Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB')"

# Benchmark quick test
benchmark:
	@echo "Running quick benchmark..."
	@$(PYTHON) -c "import numpy as np; import time; \
	data = np.random.random(1000000); \
	start = time.time(); \
	result = np.sum(data); \
	print(f'Sum of 1M elements: {time.time() - start:.4f} seconds')"

# Clean generated files
clean:
	rm -f *.json
	rm -f *.png
	rm -f *.csv
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Deep clean including virtual environment
clean-all: clean
	rm -rf $(VENV_DIR)
	rm -f $(REQUIREMENTS)

# Install system dependencies (Ubuntu/Debian)
install-system-deps:
	sudo apt-get update
	sudo apt-get install python3 python3-pip python3-venv python3-dev
	sudo apt-get install build-essential

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Check Python and install dependencies"
	@echo "  check-python     - Verify Python installation"
	@echo "  install-deps     - Install Python packages"
	@echo "  run              - Run all programs"
	@echo "  run-array        - Run parallel array operations"
	@echo "  run-matrix       - Run matrix operations"
	@echo "  run-performance  - Run performance analysis"
	@echo "  test-deps        - Test if dependencies are installed"
	@echo "  check-system     - Show system information"
	@echo "  benchmark        - Run quick benchmark"
	@echo "  clean            - Clean generated files"
	@echo "  clean-all        - Clean everything including venv"
	@echo "  help             - Show this help"

.PHONY: all check-python venv install-deps run run-array run-matrix run-performance test-deps check-system benchmark clean clean-all install-system-deps help