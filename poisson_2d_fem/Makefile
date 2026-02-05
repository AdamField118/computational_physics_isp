# Makefile for Poisson 2D FEM Solver
# Cross-platform (Windows/Linux)

# Detect OS
ifeq ($(OS),Windows_NT)
    RM = del /Q
    RMDIR = rmdir /S /Q
    MV = move
    SO_EXT = *.pyd
    NULL = 2>nul
    CONDA_LIB = $(CONDA_PREFIX)\Library\lib
else
    RM = rm -f
    RMDIR = rm -rf
    MV = mv
    SO_EXT = *.so
    NULL = 2>/dev/null
    CONDA_LIB = $(CONDA_PREFIX)/lib
endif

# Directories
FORTRAN_DIR = fortran
PYTHON_DIR = python

.PHONY: all build clean test

all: build

build:
	@echo =========================================
	@echo Building Poisson 2D FEM Solver
	@echo =========================================
	@echo.
	@echo Creating f2py wrapper...
	cd $(FORTRAN_DIR) && \
	f2py -c -m fem_fortran \
	     only: solve_poisson_2d : \
	     types_module.f90 reference_element.f90 assembly.f90 boundary_conditions.f90 solver.f90 python_interface.f90 \
	     --f90flags="-O3" \
	     -L$(CONDA_LIB) -lopenblas
	@echo.
	@echo Moving shared library...
	$(MV) $(FORTRAN_DIR)\fem_fortran$(SO_EXT) $(PYTHON_DIR)\ $(NULL) || $(MV) $(FORTRAN_DIR)/fem_fortran$(SO_EXT) $(PYTHON_DIR)/ $(NULL)
	@echo.
	@echo =========================================
	@echo Build complete!
	@echo =========================================
	@echo.
	@echo To test:
	@echo   cd python
	@echo   python convergence_study.py

clean:
	@echo Cleaning build artifacts...
	-cd $(FORTRAN_DIR) && $(RM) *.o *.mod *.a $(SO_EXT) $(NULL)
	-cd $(FORTRAN_DIR) && $(RMDIR) .mesonpy-* __pycache__ $(NULL)
	-cd $(PYTHON_DIR) && $(RM) $(SO_EXT) $(NULL)
	-cd $(PYTHON_DIR) && $(RMDIR) __pycache__ $(NULL)
	@echo Clean complete!

test:
	cd $(PYTHON_DIR) && python convergence_study.py

build+test: build test