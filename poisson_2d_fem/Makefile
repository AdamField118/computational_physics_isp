# Linux build using NumPy's Meson backend and OpenBLAS.
PYTHON ?= python3
.PHONY: all build clean test build+test
all: build
build:
	$(PYTHON) ../scripts/build.py poisson
clean:
	rm -f fortran/*.o fortran/*.mod fortran/*.so python/*.so
	rm -rf fortran/.mesonpy-* fortran/__pycache__ python/__pycache__
test:
	cd python && $(PYTHON) convergence_study.py
build+test: build test
