#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/timeb.h>

#define REAL float
#define FILTER_HEIGHT 5
#define FILTER_WIDTH 5
#define TEST 10
#define PROBLEM 256
#define PROBLEM_SIZE 768
#define TEAM_SIZE 128

// clang -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_35 --cuda-path=/usr/local/cuda -O3 -lpthread -fpermissive -msse4.1 stencil_metadirective.c -o stencil.out
// Usage: ./stencil.out <size>
// e.g. ./stencil.out 512

void Convolution(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionMulti(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionCPU(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionWorksharing(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionWorksharingUnroll(int width, int height, float fc,float fn0, float fs0, float fw0, float fe0, float fn1, float fs1, float fw1, float fe1, REAL *src, REAL *dst);
void ConvolutionMetadirective(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);

static double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

void print_array(char *title, char *name, REAL *A, int n, int m) {
    printf("%s:\n", title);
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%s[%d][%d]:%f  ", name, i, j, A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void initialize(int width, int height, REAL alpha, REAL *dx, REAL *dy, REAL *u, REAL *f_p) {
    int i;
    int j;
    int xx;
    int yy;
    int N = width*height;

    for (i = 0; i < N; i++)
        u[i] = rand() % 256;
}

int main(int argc, char *argv[]) {
    int n = PROBLEM;
    int m = PROBLEM;

    if (argc == 2) {
        n = atoi(argv[1]);
        m = atoi(argv[1]);
    };

    REAL alpha = 0.0543;
    REAL dx;
    REAL dy;

    REAL *u = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_cpu = (REAL *) malloc(sizeof(REAL) * n * m);
    initialize(n, m, alpha, &dx, &dy, u, result);

    const float filter[FILTER_HEIGHT][FILTER_WIDTH] = {
        { 0,  0, 1, 0, 0, },
        { 0,  0, 2, 0, 0, },
	{ 3,  4, 5, 6, 7, },
	{ 0,  0, 8, 0, 0, },
	{ 0,  0, 9, 0, 0, },
    };

    float fc = filter[2][2];
    float fn0 = filter[1][2];
    float fs0 = filter[3][2];
    float fw0 = filter[2][1];
    float fe0 = filter[2][3];
    float fn1 = filter[0][2];
    float fs1 = filter[4][2];
    float fw1 = filter[2][0];
    float fe1 = filter[2][4];

    int width = m;
    int height = n;
    initialize(n, m, alpha, &dx, &dy, u, result);

    // warm up the GPU
    for (int i = 0; i < 10; i++) {
        //ConvolutionWorksharingUnroll(width, height, fc,fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1, u,result);
        ConvolutionWorksharing(u, result, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    };

    double elapsed = read_timer_ms();
    //stencil(width, height, fc,fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1, u,result);
    for (int i = 0; i < TEST; i++) {
        //ConvolutionWorksharingUnroll(width, height, fc,fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1, u,result);
        //ConvolutionMetadirective(u, result, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        ConvolutionWorksharing(u, result, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    };
    elapsed = read_timer_ms() - elapsed;
    printf("GPU time(ms): %g\n", elapsed/TEST);

    elapsed = read_timer_ms();
    for (int i = 0; i < TEST; i++) {
        ConvolutionCPU(u, result_cpu, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    };
    elapsed = read_timer_ms() - elapsed;
    printf("CPU time(ms): %g\n", elapsed/TEST);

    double dif = 0;
    for (int i = 0; i < width*height; i++) {
        int x = i % width;
        int y = i / width;
        if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
            dif += fabs(result[i] - result_cpu[i]);
    }
    printf("verify dif =%f\n", dif);
    free(u);
    free(result);
    return 0;
}
void ConvolutionMetadirective(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int N = width*height; // total number of cells in a block
//#pragma omp metadirective \
    when (user={condition(N > PROBLEM_SIZE)}: /* GPU offloading */ \
        target teams distribute parallel for map(to: src[0:N], filter[0:flt_width*flt_height]) map(from: dst[0:N]) num_teams(N/TEAM_SIZE) num_threads(TEAM_SIZE)) \
    default (parallel for) /* CPU parallel */
    for (int i = 0; i < N; i++) {
        int h = i / width; // block height
        int w = i % width; // block width
        REAL sum = 0;
        for (int n = 0; n < flt_width; n++) {
            for (int m = 0; m < flt_height; m++) {
                int x = w + n - flt_width / 2;
                int y = h + m - flt_height / 2;
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    int idx = m*flt_width + n;
                    sum += src[y*width + x] * filter[idx];
                }
            }
        }
        dst[h*width + w] = sum;
    }
}


void Convolution(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height)
{
    int flt_size = flt_width*flt_height;
    int N = width*height;
    int WARP_SIZE = 32;

#pragma omp target teams map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) num_teams(N/WARP_SIZE) thread_limit(WARP_SIZE)
#pragma omp parallel num_threads(WARP_SIZE)
    {
        int localN = N/(omp_get_num_teams() * omp_get_num_threads());
        int global_thread_id = omp_get_thread_num() + omp_get_num_threads() * omp_get_team_num();
        int start = global_thread_id * localN;
        int i = global_thread_id / width;
        int j = global_thread_id % width;

        REAL sum = 0;
        for (int n = 0; n < flt_width; n++) {
            for (int m = 0; m < flt_height; m++) {
                int x = j + n - flt_width / 2;
                int y = i + m - flt_height / 2;
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    int idx = m*flt_width + n;
                    sum += src[y*width + x] * filter[idx];
                }
            }
        }
        dst[i*width + j] = sum;
        //printf("N: %d, localN: %d, ID: %d, WIDTH: %d, HEIGHT: %d, num_team: %d, num_thd: %d\n", N, localN, global_thread_id, j, i, omp_get_num_teams(), omp_get_num_threads());
    }

}


void ConvolutionMulti(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height)
{
    int flt_size = flt_width*flt_height;
    int N = width*height;
    int WARP_SIZE = 32;

#pragma omp target teams map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) num_teams(N/WARP_SIZE/4) thread_limit(WARP_SIZE)
#pragma omp parallel num_threads(WARP_SIZE)
    {
        int localN = N/(omp_get_num_teams() * omp_get_num_threads());
        int global_thread_id = omp_get_thread_num() + omp_get_num_threads() * omp_get_team_num();
        int start = global_thread_id * localN;
        int i = start / width;
        int j = start % width;

        for (int k = 0; k < localN; k++) {
            REAL sum = 0;
            for (int n = 0; n < flt_width; n++) {
                for (int m = 0; m < flt_height; m++) {
                    int x = j + k + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m*flt_width + n;
                        sum += src[y*width + x] * filter[idx];
                    }
                }
            }
            dst[i*width + j + k] = sum;
        }
        //printf("N: %d, localN: %d, ID: %d, WIDTH: %d, HEIGHT: %d, num_team: %d, num_thd: %d\n", N, localN, global_thread_id, j, i, omp_get_num_teams(), omp_get_num_threads());
    }

}

void ConvolutionWorksharing(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height)
{
	
	int flt_size = flt_width*flt_height;
    int N = width*height;
    int BLOCK_SIZE = 128;

#pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) num_teams(N/BLOCK_SIZE) num_threads(BLOCK_SIZE) collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            REAL sum = 0;
            for (int n = 0; n < flt_width; n++) {
                for (int m = 0; m < flt_height; m++) {
                    int x = j + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m*flt_width + n;
                        sum += src[y*width + x] * filter[idx];
                    }
                }
            }
            dst[i*width + j] = sum;
        }
    }
}


void ConvolutionCPU(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height)
{
    int flt_size = flt_width*flt_height;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            REAL sum = 0;
            for (int n = 0; n < flt_width; n++) {
                for (int m = 0; m < flt_height; m++) {
                    int x = j + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m*flt_width + n;
                        sum += src[y*width + x] * filter[idx];
                    }
                }
            }
            dst[i*width + j] = sum;
        }
    }
}

void ConvolutionWorksharingUnroll(int width, int height, float fc,float fn0, float fs0, float fw0, float fe0, float fn1, float fs1, float fw1, float fe1, REAL *src, REAL *dst) {
	#pragma omp target map(to: src[0:width*height], fc, fn0, fn1, fw1, fw0, fe1, fe0, fs1, fs0, height, width) map(from: dst[0:width*height])
	#pragma omp teams distribute parallel for num_teams(height) num_threads(width) collapse(2)
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			REAL sum = 0;

			sum += src[i*width+j+2] * fe1;
			sum += src[i*width+j+1] * fe0;

			sum = src[(i-2)*width+j] * fn1;
			sum += src[(i-1)*width+j] * fn0;
			sum += src[i*width+j] * fc;
			sum += src[(i+1)*width+j] * fs0;
			sum += src[(i+2)*width+j] * fs1;

			sum += src[i*width+j-1] * fw0;
			sum += src[i*width+j-2] * fw1;

			dst[i*width + j] = sum;
		}
	}
}

