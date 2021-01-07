
default: lud_parallel_for.out

all: lud_nested.out lud_parallel_for.out lud_simd.out lud_target_parallel_for.out lud_cuda.out

lud_cuda.out: lud_cuda.cu
	nvcc lud_cuda.cu -o lud_cuda.out

lud_parallel_for.out: main.o parallel_for.o
	clang main.o parallel_for.o -o lud_parallel_for.out -fopenmp

lud_simd.out: main.o simd.o
	clang main.o simd.o -o lud_simd.out -fopenmp

lud_nested.out: main.o parallel_for_simd_nested.o
	clang main.o parallel_for_simd_nested.o -o lud_nested.out -fopenmp

lud_target_parallel_for.out: main.c target_parallel_for.c
	clang -fopenmp -fopenmp-targets=nvptx64 main.c target_parallel_for.c -o lud_target_parallel_for.out

main.o: main.c
	clang -c main.c -o main.o

simd.o: simd.c
	clang -c simd.c -o simd.o -fopenmp -mavx2

parallel_for.o: parallel_for.c
	clang -c parallel_for.c -o parallel_for.o -fopenmp

parallel_for_simd_nested.o: parallel_for_simd_nested.c
	clang -c parallel_for_simd_nested.c -o parallel_for_simd_nested.o -fopenmp -mavx2

target_parallel_for.o: target_parallel_for.c
	clang -c target_parallel_for.c -o target_parallel_for.o -fopenmp -fopenmp-targets=nvptx64

clean:
	rm -rf *.out *.o *.log *.csv
