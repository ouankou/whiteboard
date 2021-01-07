
default: lud_parallel_for.out

all: lud_nested.out lud_parallel_for.out lud_simd.out lud_target_parallel_for.out lud_cuda.out

lud_cuda.out: lud_cuda.cu
	nvcc lud_cuda.cu -o lud_cuda.out

lud_parallel_for.out: main.o parallel_for.o
	xlc main.o parallel_for.o -o lud_parallel_for.out -qsmp=omp

lud_simd.out: main.o simd.o
	xlc main.o simd.o -o lud_simd.out -qsmp=omp

lud_nested.out: main.o parallel_for_simd_nested.o
	xlc main.o parallel_for_simd_nested.o -o lud_nested.out -qsmp=omp

lud_target_parallel_for.out: main.c target_parallel_for.c
	xlc main.c target_parallel_for.c -o lud_target_parallel_for.out -qsmp=omp -qoffload

main.o: main.c
	xlc -c main.c -o main.o

simd.o: simd.c
	xlc -c simd.c -o simd.o -qsmp=omp

parallel_for.o: parallel_for.c
	xlc -c parallel_for.c -o parallel_for.o -qsmp=omp

parallel_for_simd_nested.o: parallel_for_simd_nested.c
	xlc -c parallel_for_simd_nested.c -o parallel_for_simd_nested.o -qsmp=omp

target_parallel_for.o: target_parallel_for.c
	xlc -c target_parallel_for.c -o target_parallel_for.o -qsmp=omp -qoffload

clean:
	rm -rf *.out *.o *.log
