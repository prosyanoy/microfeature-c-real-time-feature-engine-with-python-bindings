# Convenience wrapper around CMake. `make setup build test` is the happy path.
VENV   ?= .venv
PY     := $(VENV)/bin/python
CMAKE  := $(VENV)/bin/cmake
JOBS   ?= $(shell nproc)
GEN    := -G Ninja -DCMAKE_MAKE_PROGRAM=$(abspath $(VENV))/bin/ninja
PYB    := -DPython3_EXECUTABLE=$(abspath $(PY)) -Dpybind11_DIR=$(shell $(PY) -m pybind11 --cmakedir 2>/dev/null)

.PHONY: chart showcase setup build test test-cpp test-py asan tsan bench bench-cpp demo perf clean

setup:
	python3 -m venv --system-site-packages $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt

build:
	$(CMAKE) -S . -B build $(GEN) $(PYB)
	$(CMAKE) --build build -j $(JOBS)

test: test-cpp test-py

test-cpp:
	cd build && $(abspath $(VENV))/bin/ctest --output-on-failure

test-py:
	MF_MODULE_DIR=$(abspath build/python) $(PY) -m pytest tests/python -q

# Sanitizer builds skip the Python module (a sanitized runtime can't be dlopen()ed into python).
asan:
	$(CMAKE) -S . -B build-asan $(GEN) -DMF_SANITIZER=address -DMF_BUILD_BENCH=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
	$(CMAKE) --build build-asan -j $(JOBS)
	cd build-asan && ASAN_OPTIONS=detect_leaks=1 $(abspath $(VENV))/bin/ctest --output-on-failure

tsan:
	$(CMAKE) -S . -B build-tsan $(GEN) -DMF_SANITIZER=thread -DMF_BUILD_BENCH=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
	$(CMAKE) --build build-tsan -j $(JOBS)
	cd build-tsan && TSAN_OPTIONS="halt_on_error=1" $(abspath $(VENV))/bin/ctest --output-on-failure

bench-cpp:
	./build/microfeature_bench

bench:
	MF_MODULE_DIR=$(abspath build/python) $(PY) benchmarks/run_benchmarks.py

demo:
	MF_MODULE_DIR=$(abspath build/python) $(PY) examples/ml_demo.py

perf:
	scripts/perf_profile.sh

clean:
	rm -rf build build-asan build-tsan

chart:
	$(PY) benchmarks/plot_results.py

showcase:
	scripts/demo_45s.sh
