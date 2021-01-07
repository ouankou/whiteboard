#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <sys/timeb.h>

#define NUM_RUNS 100

// read timer in second
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

double* make2d(long n) {
    double *m;
    m = (double *) malloc(n * n * sizeof(double *));
    return m;
}

void initializeVersion2(double *A,long n){
	long i,j, k;
	for(i=0;i<n;i++){
		for(j=i;j<n;j++){
			if(i==j){
				k=i+1;
				A[i*n+j]=4*k-3;
			}
			else{
				A[i*n+j]=A[i*n+i]+1;
				A[j*n+i]=A[i*n+i]+1;
			}
		}
	}
}

void printmatrix(double *A, long n) {
    printf("\n *************** MATRIX ****************\n\n");
    long i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            printf("%f ",A[i*n+j]);
        printf("\n");
    }
}

int checkVersion2(double *A, long n)
{
	long i,j;
	for(i=0;i<n;i++){
		if(A[i*n+i]!=1){
			return 0;
		}
		for(j=0;j<n;j++){
			if(i!=j && A[i*n+j]!=2){
				return 0;
			}
		}
	}
	return 1;
}

__global__ void lud_kernel(double *matrix, long size, int k){
    int row = blockIdx.x;
    int col = threadIdx.x;
    /*
#pragma omp parallel for
        for (row = k + 1; row < size; row++) {
            matrix[row*size+k] /= matrix[k*size+k];
        };
    */
    if (row > k && col == 0) {
        matrix[row * size + k] /= matrix[k * size + k];
    }
    __syncthreads();
    /*
#pragma omp parallel for private(row) shared(matrix)
        for (row = k + 1; row < size; row++) {
            int col = 0;
            double factor = matrix[row*size+k];
            for (col = k + 1; col < size; col++) { //column
                matrix[row*size+col] = matrix[row*size+col] - factor*matrix[k*size+col];
            }
            matrix[row*size+k] = factor;
        }
    */
    if (row > k && col > k) {
        matrix[row*size+col] = matrix[row*size+col] - matrix[row*size+k]*matrix[k*size+col];
    };
    return;
}

int main(int argc, char** argv) {
    int n = 256;
    if (argc > 1) {
        n = atol(argv[1]);
    };
    if (n > 1024) {
        printf("Currently, n can only be up to 1024.\n");
        return 1;
    };
    int i;
    double* matrix = NULL;
    double* dev_matrix = NULL;

    double start = read_timer();

    for (i = 0; i < NUM_RUNS; i++) {
        matrix = make2d(n);
        initializeVersion2(matrix, n);
        size_t cap = n * n * sizeof(double);
        int num_blocks = n, threads_per_block = n, k;

        cudaMalloc((void**)&dev_matrix, cap);
        cudaMemcpy(dev_matrix, matrix, cap, cudaMemcpyHostToDevice);
        // LUD outer loop
        for (k = 0; k < n - 1; k++) {
            lud_kernel<<<num_blocks, threads_per_block>>>(dev_matrix, n, k);
        };
        cudaDeviceSynchronize();
        cudaMemcpy(matrix, dev_matrix, cap, cudaMemcpyDeviceToHost);
    }

    double total_time = read_timer() - start;

    fprintf(stderr, "%s", checkVersion2(matrix, n)==1? "DECOMPOSE SUCCESSFULL\n":"DECOMPOSE FAIL\n");
    printf("%lg", total_time);

    cudaFree(dev_matrix);
    free(matrix);
    return 0;
};
