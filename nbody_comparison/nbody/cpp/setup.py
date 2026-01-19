"""
Setup script for C++ N-body module using pybind11

To build:
    pip install pybind11
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'nbody_cpp_module',
        ['nbody.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11'],
    ),
]

setup(
    name='nbody_cpp_module',
    ext_modules=ext_modules,
)