# Makefile for FEM 1D Benchmark

.PHONY: all build clean test benchmark dashboard help

all: build

help:
	@echo "FEM 1D Benchmark - Available targets:"
	@echo "  make build      - Build all implementations"
	@echo "  make test       - Run quick correctness test"
	@echo "  make benchmark  - Run full benchmark suite"
	@echo "  make dashboard  - Generate web dashboard"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make all        - Build everything (default)"

build:
	@echo "Building all implementations..."
	bash ./build.sh

test:
	@echo "Running correctness tests..."
	cd benchmark && python3 -c "from benchmark import BenchmarkSuite; s = BenchmarkSuite(); s.load_implementations(); s.verify_correctness()"

benchmark: build
	@echo "Running full benchmark suite..."
	cd benchmark && python3 benchmark.py

dashboard: benchmark
	@echo "Generating web dashboard..."
	cd benchmark && python3 visualize.py
	@echo "Dashboard ready: web/benchmark_results.html"

clean:
	@echo "Cleaning build artifacts..."
	rm -f c/*.so
	rm -f cpp/*.so
	rm -f fortran/*.so
	rm -rf rust/target
	rm -f results/*.json
	rm -f web/benchmark_results.html
	@echo "Clean complete"

# Individual language builds
c:
	@echo "Building C implementation..."
	cd c && gcc -O3 -fPIC -shared -fopenmp -o fem_c.so fem_assembly.c -lgomp

cpp:
	@echo "Building C++ implementation..."
	cd cpp && c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp \
		$$(python3 -m pybind11 --includes) \
		fem_assembly.cpp -o fem_cpp$$(python3-config --extension-suffix) -lgomp

fortran:
	@echo "Building Fortran implementation..."
	cd fortran && f2py -c -m fem_fortran fem_assembly.f90 --f90flags="-fopenmp" -lgomp

rust:
	@echo "Building Rust implementation..."
	cd rust && maturin develop --release

julia:
	@echo "Setting up Julia..."