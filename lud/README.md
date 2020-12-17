# Introduction

There are five versions of LU decomposition.

1. OpenMP parallel for
2. OpenMP simd
3. Nested OpenMP parallel for and simd
4. OpenMP target parallel for
5. Native CUDA

The first four versions work as a external library and share the same `main.c`.
The CUDA version is standalone and used to show the performance difference between CUDA and OpenMP implementation.

# Environment

The code has been tested with Clang/LLVM 10.x and CUDA toolkit 10.1 on Ubuntu 18.04.

# Build

```bash
make all
```

# Run

```bash
# <executable> <problem size>
./lud_cuda.out 1024
```

# Others

Please notice that the CUDA version only support the problem size up to 1024 due to the initial implementation.
