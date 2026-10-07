PYTHON ?= python3
.PHONY: all build clean
all: build
build:
	$(PYTHON) ../../../scripts/build.py nbody --language c
clean:
	rm -f *.so *.o *.mod
