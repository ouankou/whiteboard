#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/timeb.h>

#define REAL float
#define FILTER_HEIGHT 5
#define FILTER_WIDTH 5
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

void initialize(int n, int m, REAL alpha, REAL *dx, REAL *dy, REAL *u_p, REAL *f_p) {
    int i;
    int j;
    int xx;
    int yy;
    REAL (*u)[m] = (REAL (*)[m]) u_p;
    REAL (*f)[m] = (REAL (*)[m]) f_p;
    *dx = (2.0 / (n - 1));
    *dy = (2.0 / (m - 1));
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++) {
            xx = ((int) (-1.0 + (*dx * (i - 1))));
            yy = ((int) (-1.0 + (*dy * (j - 1))));
            u[i][j] = 0.0;
            f[i][j] = (((((-1.0 * alpha) * (1.0 - (xx * xx)))
                         * (1.0 - (yy * yy))) - (2.0 * (1.0 - (xx * xx))))
                       - (2.0 * (1.0 - (yy * yy))));
        }
}

void stencil(int n, int m, float fc,float fn0, float fs0, float fw0, float fe0, float fn1, float fs1, float fw1, float fe1, REAL * u, REAL * result);

int main(int argc, char *argv[]) {
    int n = 512;
    int m = 512;
    REAL alpha = 0.0543;
    REAL dx;
    REAL dy;

    REAL *u = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result = (REAL *) malloc(sizeof(REAL) * n * m);
    initialize(n, m, alpha, &dx, &dy, u, result);

    float filter[FILTER_HEIGHT][FILTER_WIDTH] = {
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

    initialize(n, m, alpha, &dx, &dy, u, result);

    double elapsed = read_timer_ms();
    stencil(n, m, fc,fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1, u,result);
    elapsed = read_timer_ms() - elapsed;
    printf("time(ms): %12.6g\n", elapsed);
    free(u);
    free(result);
    return 0;
}

void stencil(int n, int m, float fc,float fn0, float fs0, float fw0, float fe0, float fn1, float fs1, float fw1, float fe1, REAL * u, REAL * result) {
    int i, j, k;

    //int FILTER_WIDTH = 5;
    //int FILTER_HEIGHT = 5;
    int width = m;
    #pragma omp target map(u,result)
    #pragma omp parallel for
    for(int i = FILTER_WIDTH / 2; i <= n - FILTER_WIDTH / 2; i++) {
        for(int j = FILTER_HEIGHT / 2; j <= m - FILTER_HEIGHT / 2; j++){
            float sum = u[width*j+i] * fc;
            sum += u[width*j+i+1] * fe0;
            sum += u[width*j+i+2] * fe1;
            sum += u[width*j+i-1] * fw0;
            sum += u[width*j+i-2] * fw1;
            sum += u[width*(j-1)+i] * fn0;
            sum += u[width*(j-2)+i] * fn1;
            sum += u[width*(j+1)+i] * fs0;
            sum += u[width*(j+2)+i] * fs1;
	    result[width*j+i] = sum;

        }
    }
}
