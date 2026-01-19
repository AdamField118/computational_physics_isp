"""
Setup script for building Fortran N-body module with f2py

To build:
    python setup.py build_ext --inplace

Or use f2py directly:
    f2py -c nbody.f90 -m nbody_fortran_module
    
With OpenMP support (for parallel acceleration computation):
    f2py -c nbody.f90 -m nbody_fortran_module --f90flags="-fopenmp" -lgomp
"""

from numpy.distutils.core import setup, Extension
import os

# Module name that Python will import
module_name = 'nbody_fortran_module'

# Fortran source file
fortran_source = 'nbody.f90'

# Extension configuration
ext_modules = [
    Extension(
        name=module_name,
        sources=[fortran_source],
        extra_f90_compile_args=['-O3', '-fopenmp'],  # Optimization + OpenMP
        extra_link_args=['-lgomp'],  # Link OpenMP library
    )
]

if __name__ == '__main__':
    setup(
        name=module_name,
        version='1.0',
        description='Fortran N-body simulation with f2py wrapper',
        author='Adam Field',
        ext_modules=ext_modules,
    )