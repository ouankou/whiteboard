#include <stdio.h>
#include <omp.h>

// clang -fopenmp omp_parallel_for_scheduling.c -o schedule.out
int main (int argc, char** argv) {

    // t0: 0,4,8, t1: 1,5,9, t2: 2,6,10, t3:3,7,11
    printf("Cyclic scheduling.\n");
#pragma omp parallel for num_threads(4) schedule(static, 1)
    for (int i = 0; i < 12; i++)
        printf("Thread ID: %d, Iteration: %d\n", omp_get_thread_num(), i);

    // t0: 0,1,2, t1: 3,4,5, t2: 6,7,8, t3:9,10,11
    printf("Block scheduling.\n");
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < 12; i++)
        printf("Thread ID: %d, Iteration: %d\n", omp_get_thread_num(), i);

    return 0;
}
