# Makefile for FEM 1D Benchmark

.PHONY: all build clean test benchmark plots help

all: build

help:
	@echo "FEM 1D Benchmark - Available targets:"
	@echo "  make build      - Build all implementations"
	@echo "  make test       - Run quick correctness test"
	@echo "  make benchmark  - Run full benchmark suite"
	@echo "  make plots      - Generate static figures and Markdown tables"
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

plots:
	@echo "Generating static figures and Markdown tables..."
	cd benchmark && python3 visualize.py
	@echo "Results: results/assembly_comparison.png and results/benchmark_summary.md"

clean:
	@echo "Cleaning build artifacts..."
	rm -f c/*.so
	rm -f cpp/*.so
	rm -f fortran/*.so fortran/*.mod fortran/*.o
	rm -f rust/*.so
	rm -rf rust/target
	@echo "Clean complete"

# Individual language builds
.PHONY: c cpp fortran rust julia
c cpp fortran rust:
	python3 ../scripts/build.py fem --language $@

julia:
	python3 ../scripts/build.py julia
