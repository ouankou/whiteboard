// Experimental test input for Accelerator directives
//  simplest scalar*vector operations
// Liao 1/15/2013
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "sum.h"
#define TEST 20

double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL double
/* zero out the entire vector */
void zero(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a vector with random floating point numbers */
void init(REAL *A, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = (REAL)drand48();
    }
}

/*serial version */
REAL sum_serial(REAL* input, int n) {
    int i;
    REAL sum = 0.0;
    for (i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

/*OpenMP CPU version */
REAL sum_omp_cpu(REAL* input, int n) {
    int i;
    REAL sum = 0.0;
#pragma omp parallel for reduction(+: sum)
    for (i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

/* compare two arrays and return percentage of difference */
REAL check(REAL*A, REAL*B, int n)
{
    int i;
    REAL diffsum = 0.0, sum = 0.0;
    for (i = 0; i < n; i++) {
        diffsum += fabs(A[i] - B[i]);
        sum += fabs(B[i]);
    }
    return diffsum/sum;
}

int main(int argc, char *argv[]) {
    int n = 512;
    int kernel = 0;
    REAL *output_device, *output, *input, sum;

    if (argc > 2) {
        n = atoi(argv[1]);
        kernel = atoi(argv[2]);
    }
    else {
        printf("Usage: ./sum <n>\n");
        printf("Default size: n = 512\n");
    };
    output_device = (REAL *) malloc(n * sizeof(REAL));
    output = (REAL *) malloc(n * sizeof(REAL));
    input = (REAL *) malloc(n * sizeof(REAL));

    srand48(1<<12);
    init(input, n);

    REAL res_serial = sum_serial(input, n);
    REAL res_omp_cpu = sum_omp_cpu(input, n);

    printf("CPU serial vs omp: %g, %g, %g\n", res_serial, res_omp_cpu, res_serial - res_omp_cpu);

    int i;
    /* cuda version */
    if (kernel == -1) {
        for (i = 0; i < TEST; i++) {
            sum_kernel(input, n, 0);
            sum_kernel(input, n, 1);
            sum_kernel(input, n, 2);
        };
        free(output_device);
        free(output);
        free(input);
        return 0;
    };

    REAL res_cuda;
    /*
    // warm up
    for (i = 0; i < 10; i++) {
        res_cuda = sum_kernel(input, n, kernel);
    };
    */
    double elapsed = read_timer_ms();
    for (i = 0; i < TEST; i++) {
        res_cuda = sum_kernel(input, n, kernel);
    };
    elapsed = (read_timer_ms() - elapsed)/TEST;
    printf("GPU kernel %d: %g\n", kernel, elapsed);
    printf("CPU omp vs GPU: %g, %g, %g\n", res_omp_cpu, res_cuda, res_cuda - res_omp_cpu);

    //REAL checkresult = check(res_serial, output_device, n);
    //printf("axpy(%d): checksum: %g, time: %0.2fms\n", n, checkresult, elapsed);
    //assert (checkresult < 1.0e-10);

    free(output_device);
    free(output);
    free(input);
    return 0;
}
