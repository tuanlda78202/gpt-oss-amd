CC := $(shell command -v hipcc 2>/dev/null || echo g++)
CFLAGS = --std=c++17 -lm
ifneq ($(CC),g++)
CFLAGS += --offload-arch=gfx90a
endif

CPP_FILES = src/run.cpp src/tokenizer.cpp

.PHONY: run
run: $(CPP_FILES) tokenizer-bin
	$(CC) -g -O0 -o build/run $(CPP_FILES)

rundebug: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -g -o build/run $(CPP_FILES)

.PHONY: runfast
runfast: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -o build/run $(CPP_FILES)

.PHONY: runomp
runomp: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -fopenmp -march=native $(CPP_FILES) -o build/run

.PHONY: decode
decode: src/decode.cpp src/tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -O3 src/decode.cpp src/tokenizer.cpp -o build/decode

.PHONY: tokenizer-bin
tokenizer-bin: tools/export_tokenizer_bin.py
	python3 tools/export_tokenizer_bin.py -o build/tokenizer.bin

.PHONY: tokenizer-test
tokenizer-test: tests/test_tokenizer.cpp src/tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -DTESTING -O3 tests/test_tokenizer.cpp src/tokenizer.cpp -o build/test_tokenizer

.PHONY: clean
clean:
	rm -f build/run build/decode build/tokenizer.bin build/test_tokenizer
