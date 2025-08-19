# Use hipcc by default.
# If hipcc isn't available, fall back to g++.
CC := $(shell command -v hipcc 2>/dev/null || echo g++)
CFLAGS = --std=c++17 -lm
ifneq ($(CC),g++)
CFLAGS += --offload-arch=gfx90a
endif

CPP_FILES = src/run.cpp src/tokenizer.cpp

# Basic build that should work on most systems.
.PHONY: run
run: $(CPP_FILES) tokenizer-bin
	$(CC) -g -O0 -o build/run $(CPP_FILES)

# Debug build; suitable for tools like Valgrind. Example:
#   valgrind --leak-check=full ./build/run out/model.bin -n 3
rundebug: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -g -o build/run $(CPP_FILES)

# Optimized build (-O3). See GCC optimization options:
# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
.PHONY: runfast
runfast: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -o build/run $(CPP_FILES)

# Build with OpenMP for multithreaded runs.
# Remember to set threads when running, e.g.:
#   OMP_NUM_THREADS=4 ./build/run out/model.bin
.PHONY: runomp
runomp: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -fopenmp -march=native $(CPP_FILES) -o build/run

# Build the 'decode' utility.
.PHONY: decode
decode: src/decode.cpp src/tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -O3 src/decode.cpp src/tokenizer.cpp -o build/decode

# Generate tokenizer.bin using the Python exporter.
.PHONY: tokenizer-bin
tokenizer-bin: tools/export_tokenizer_bin.py
	python3 tools/export_tokenizer_bin.py -o build/tokenizer.bin

# Build the tokenizer test binary (defines TESTING).
.PHONY: tokenizer-test
tokenizer-test: tests/test_tokenizer.cpp src/tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -DTESTING -O3 tests/test_tokenizer.cpp src/tokenizer.cpp -o build/test_tokenizer

# Remove build artifacts.
.PHONY: clean
clean:
	rm -f build/run build/decode build/tokenizer.bin build/test_tokenizer
