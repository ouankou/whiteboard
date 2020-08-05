#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/timeb.h>
#include <pthread.h>
#include <float.h>

#define REAL float
#define FILTER_HEIGHT 5
#define FILTER_WIDTH 5
#define TEST 200
#define PROBLEM 256
#define PROBLEM_SIZE 768
#define TEAM_SIZE 128
#define PROFILE_NUM 20

// clang -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_35 --cuda-path=/usr/local/cuda -O3 -lpthread -fpermissive -msse4.1 stencil_metadirective.c -o stencil.out
// Usage: ./stencil.out <size>
// e.g. ./stencil.out 512

void Convolution(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionMulti(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionCPU(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionWorksharing(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ConvolutionWorksharingUnroll(int width, int height, float fc,float fn0, float fs0, float fw0, float fe0, float fn1, float fs1, float fw1, float fe1, REAL *src, REAL *dst);
void ConvolutionMetadirective(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void CPM(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void FPM(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void ODDC(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void* ODDC_CPU(void* arguments);
void* ODDC_GPU(void* arguments);

// variables for CPM 
int cpm_cpu_test = 0;
int cpm_gpu_test = 0;
double cpm_cpu_speed = 0.0;
double cpm_gpu_speed = 0.0;

// oddc time
double oddc_cpu_time = 0.0;
double oddc_gpu_time = 0.0;

// 0 for CPU, 1 for GPU
double checkFPM(int N, int device) {
    double result = 0;
    // Lassen
    //if (device == 0) { result = 11565 + 1983 * log(N); }
    //else { result = -30958 + 9961 * log(N); }
    
    // Pascal
    if (device == 0) { result = 15581 + 1571 * log(N); }
    else { if (N < 200) {result = -270 + 166*N - 0.103 * N * N; } else result = -76839 + 21096 * log(N); }
    
    // Surface
    //if (device == 0) { result = -2576 + 3610 * log(N); }
    //else { result = -20039 + 7341 * log(N); }

    return result;
}


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
    REAL *result_cpm = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_fpm = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_oddc = (REAL *) malloc(sizeof(REAL) * n * m);
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

    // warm up the functions
    for (int i = 0; i < 10; i++) {
        ConvolutionWorksharing(u, result, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        ConvolutionCPU(u, result_cpu, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        CPM(u, result_cpm, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        FPM(u, result_fpm, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        ODDC(u, result_oddc, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    };
    cpm_cpu_test = 0;
    cpm_gpu_test = 0;
    cpm_cpu_speed = 0.0;
    cpm_gpu_speed = 0.0;

    double elapsed = read_timer_ms();
    // Reset the CPM model after warming up
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    double cpm_time = 0.0;
    double fpm_time = 0.0;
    double oddc_time = 0.0;
    oddc_cpu_time = 0.0;
    oddc_gpu_time = 0.0;
    double dif = 0.0;

    //for (int k = 16; k < width; k *= 2) { // fpm test loop start
        //int width2 = k;
        //int height2 = k;
    for (int k = 0; k < 1; k++) { // fpm test loop start
        int width2 = width;
        int height2 = height;
    	//printf("%d\n", k);
    for (int i = 0; i < TEST; i++) {
    	elapsed = read_timer_ms();
        ConvolutionWorksharing(u, result, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        gpu_time += read_timer_ms() - elapsed;
    	elapsed = read_timer_ms();
        ConvolutionCPU(u, result_cpu, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        cpu_time += read_timer_ms() - elapsed;
    	elapsed = read_timer_ms();
        CPM(u, result_cpm, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        cpm_time += read_timer_ms() - elapsed;
    	elapsed = read_timer_ms();
        FPM(u, result_fpm, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        fpm_time += read_timer_ms() - elapsed;
    	elapsed = read_timer_ms();
        ODDC(u, result_oddc, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        oddc_time += read_timer_ms() - elapsed;
        dif = 0.0;
    for (int j = 0; j < width2*height2; j++) {
        int x = j % width2;
        int y = j / width2;
        if (x > FILTER_WIDTH/2 && x < width2 - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height2 - FILTER_HEIGHT/2)
            dif += fabs(result[j] - result_cpu[j]);
            dif += fabs(result_fpm[j] - result_oddc[j]);
    }
        //printf("verify dif =%g\n", dif);
        //if (dif != 0.0) printf("verify dif =%g\n", dif);
        if (dif < -1) printf("verify dif =%g\n", dif);
        //memcpy(u, result_cpu, width*height*sizeof(REAL));
        initialize(n, m, alpha, &dx, &dy, u, result);
    };
    	//printf("%d,%g,", k, cpu_time/TEST);
    	//printf("%g\n", gpu_time/TEST);
    }; // fpm test loop end

    printf("CPU time(ms): %g\n", cpu_time/TEST);
    printf("CPU total time(ms): %g\n", cpu_time);
    printf("GPU time(ms): %g\n", gpu_time/TEST);
    printf("GPU total time(ms): %g\n", gpu_time);
    printf("CPM time(ms): %g\n", cpm_time/TEST);
    printf("CPM total time(ms): %g\n", cpm_time);
    printf("FPM time(ms): %g\n", fpm_time/TEST);
    printf("FPM total time(ms): %g\n", fpm_time);
    printf("ODDC time(ms): %g\n", oddc_time/TEST);
    printf("ODDC total time(ms): %g\n", oddc_time);
    printf("ODDC CPU time(ms): %g\n", oddc_cpu_time/TEST);
    printf("ODDC total CPU time(ms): %g\n", oddc_cpu_time);
    printf("ODDC GPU time(ms): %g\n", oddc_gpu_time/TEST);
    printf("ODDC total GPU time(ms): %g\n", oddc_gpu_time);

/*
    double dif = 0;
    for (int i = 0; i < width*height; i++) {
        int x = i % width;
        int y = i / width;
        if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
            dif += fabs(result[i] - result_cpu[i]);
    }
    printf("verify dif =%f\n", dif);
*/
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

#pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) collapse(2)
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


void ConvolutionCPU(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {

#pragma omp parallel for collapse(2) num_threads(8)
//#pragma omp parallel for collapse(2)
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

void CPM(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    double elapsed = read_timer_ms();
    int N = width*height;
    if (cpm_cpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        ConvolutionCPU(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        cpm_cpu_test += 1;
        cpm_cpu_speed += elapsed;
        if (cpm_cpu_test == PROFILE_NUM) { cpm_cpu_speed = N*N*PROFILE_NUM/cpm_cpu_speed; }
    } else if (cpm_gpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        ConvolutionWorksharing(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        cpm_gpu_test += 1;
        cpm_gpu_speed += elapsed;
        if (cpm_gpu_test == PROFILE_NUM) { cpm_gpu_speed = N*N*PROFILE_NUM/cpm_gpu_speed; }
    } else {
        if (cpm_cpu_speed > cpm_gpu_speed) { ConvolutionCPU(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT); }
        else ConvolutionWorksharing(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
    };
}

void FPM(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int N = width*height;
    double cpu_speed = checkFPM(width, 0);
    double gpu_speed = checkFPM(width, 1);
    if (gpu_speed < cpu_speed) {
        ConvolutionCPU(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
    } else {
        ConvolutionWorksharing(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
    };
}

struct arg_struct {
    const REAL* src;
    REAL* dst;
    int width;
    int height;
    const float* filter;
    int flt_width;
    int flt_height;
};



void* ODDC_GPU(void* arguments)
{
    struct arg_struct *args = (struct arg_struct *)arguments;
    const REAL* src = args->src;	
    REAL* dst = args->dst;	
    int width = args->width;	
    int height = args->height;	
    const float* filter = args->filter;	
    int flt_width = args->flt_width;	
    int flt_height = args->flt_height;	
    int flt_size = flt_width*flt_height;
    int N = width*height;

#pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) collapse(2)
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


void* ODDC_CPU(void* arguments) {

    struct arg_struct *args = (struct arg_struct *)arguments;
    const REAL* src = args->src;	
    REAL* dst = args->dst;	
    int width = args->width;	
    int height = args->height;	
    const float* filter = args->filter;	
    int flt_width = args->flt_width;	
    int flt_height = args->flt_height;	
    int flt_size = flt_width*flt_height;
#pragma omp parallel for collapse(2) num_threads(8)
//#pragma omp parallel for collapse(2)
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



void ODDC(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    int N = width*height;

    struct arg_struct args1;
    args1.src = src;
    args1.dst = dst;
    args1.width = width;
    args1.height = height;
    args1.filter = filter;
    args1.flt_width = flt_width;
    args1.flt_height = flt_height;

    struct arg_struct args2;
    args2.src = src;
    args2.dst = dst;
    args2.width = width;
    args2.height = height;
    args2.filter = filter;
    args2.flt_width = flt_width;
    args2.flt_height = flt_height;

    double time_diff[11];
    double cpu_speed = checkFPM(width, 0);
    if (cpu_speed < 1) cpu_speed = 1;
    double gpu_speed = checkFPM(width, 1);
    if (gpu_speed < 1) gpu_speed = 0.5;
    //time_diff[0] = N/checkFPM(width, 1);
    //time_diff[10] = N/checkFPM(width, 0);
    time_diff[10] = N/cpu_speed;
    int i;
    for (i = 1; i < 10; i++) {
        int cpu_share = i*N/10;
        int gpu_share = (10-i)*N/10;
        int cpu_size = sqrt(cpu_share);
        int gpu_size = height - cpu_size;
        cpu_speed = checkFPM(cpu_size, 0);
        if (cpu_speed < 1) cpu_speed = 1;
        gpu_speed = checkFPM(gpu_size, 1);
        if (gpu_speed < 1) gpu_speed = 0.5;
        //time_diff[i] = fabs(cpu_share / checkFPM(cpu_size, 0) - gpu_share / checkFPM(gpu_size, 1));
        time_diff[i] = fabs(cpu_share / cpu_speed - gpu_share / gpu_speed);
    };
    double min_diff = DBL_MAX;
    int min_index = 0;
    for (i = 0; i < 11; i++) {
         if (time_diff[i] < min_diff) {
             min_diff = time_diff[i];
             min_index = i;
         };
    };
    //min_index = 2.5;
    //printf("Best ratio: %d\n", min_index);
    int height1 = min_index*height/10;
    int height2 = height - height1;
    args1.height = height1;
    args2.height = height2;
    args2.src += height1 * width;
    args2.dst += height1 * width;

    double elapsed = read_timer_ms();
    double cpu_start_time, gpu_start_time;
    pthread_t tid[2];
    gpu_start_time = read_timer_ms();
    if (height2 != 0) pthread_create(&tid[1], NULL, ODDC_GPU, (void *)&args2);
    cpu_start_time = read_timer_ms();
    if (height1 != 0) pthread_create(&tid[0], NULL, ODDC_CPU, (void *)&args1);
    if (height2 != 0) pthread_join(tid[1], NULL);
    oddc_gpu_time += read_timer_ms() - gpu_start_time;
    if (height1 != 0) pthread_join(tid[0], NULL);
    oddc_cpu_time += read_timer_ms() - cpu_start_time;

}

