PYTHON ?= python3
.PHONY: all build clean
all: build
build:
	$(PYTHON) ../../../scripts/build.py nbody --language cpp
clean:
	rm -f *.so *.o *.mod
