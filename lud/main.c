#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/timeb.h>

double* make2dmatrix(long n);
void free2dmatrix(double *M, long n);
void printmatrix(double *A, long n);
extern void decompose(double*, long);

#define NUM_RUNS 100

long matrix_size,version;
char algo;

// read timer in second
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

int checkVersion1(double *A, long n)
{
	long i, j;
	for (i=0;i<n;i++)
	{
		for (j=0;j<n;j++)
			if(A[i*n+j]!=1){
				return 0;
			}
	}
	return 1;
}

void initializeVersion1(double *A, long n)
{
	long i, j;
	for (i=0;i<n;i++){
		for (j=0;j<n;j++){
			if(i<=j )
				A[i*n+j]=i+1;
			else
				A[i*n+j]=j+1;

		}
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


double *getMatrix(long size,long version)
{
	double *m = make2dmatrix(size);
	switch(version){
	case 1:
		initializeVersion1(m, size);
		break;
	case 2:
		initializeVersion2(m, size);
		break;
	default:
		printf("INVALID VERSION NUMBER");
		exit(0);
	}
	return m;
}

int check(double *A, long size, long version){
	switch(version){
	case 1:
		return checkVersion1(A,size);
		break;
	case 2:
		return checkVersion2(A,size);
		break;
	default:
		printf("INVALID VERSION CHARACTER IN CHECK");
		exit(0);
	}
}

int main(int argc, char *argv[]){

    matrix_size = 128;
    version = 2;
    int num_threads = 4;
	if (argc > 2) {
		num_threads = atoi(argv[2]);
	};
	if (argc > 1) {
		matrix_size = atol(argv[1]);
	};
	//omp_set_num_threads(num_threads);

	//double* matrix = getMatrix(matrix_size,version);

	//printmatrix(matrix,matrix_size);
    int i;
    double* matrix = NULL;
	/**
	 * Code to Time the LU decompose
	 */
	//clock_t begin, end;
	//double time_spent;
	//begin = clock();
    double start = read_timer();

    for (i = 0; i < NUM_RUNS; i++) {
	    matrix = getMatrix(matrix_size,version);
	    decompose(matrix, matrix_size);
    };

    double total_time = read_timer() - start;
	//end = clock();
	//time_spent = ((double)(end - begin)) / CLOCKS_PER_SEC;

	//printmatrix(matrix,matrix_size);

	//printf("\n**********************************\n\n");
	//printf("Algo selected :%s\n","OpenMP");
	//printf("Size of Matrix :%lu \n",matrix_size);
	//printf("Version Number : %lu\n",version);
	fprintf(stderr, "%s",check(matrix,matrix_size,version)==1? "DECOMPOSE SUCCESSFULL\n":"DECOMPOSE FAIL\n");
	//printf("DECOMPOSE TIME TAKEN : %lg seconds\n", total_time);
	//printf("\n**********************************\n\n");
	printf("%lg", total_time);

	free2dmatrix(matrix, matrix_size);
	return 0;
}


double* make2dmatrix(long n)
{
	long i;
	double *m;
	m = (double*)malloc(n*n*sizeof(double*));
	return m;
}

// only works for dynamic arrays:
void printmatrix(double *A, long n)
{
	printf("\n *************** MATRIX ****************\n\n");
	long i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			printf("%f ",A[i*n+j]);
		printf("\n");
	}
}

void free2dmatrix(double *M, long n)
{
	long i;
	if (!M) return;
	free(M);
}
