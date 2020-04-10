#include "common.h"
#include <dlfcn.h>

#define REAL double

void j2d9pt_cpu(const REAL* __restrict__ src, REAL* dst, int width, int height, REAL fc, REAL fn0, REAL fs0, REAL fw0, REAL fe0, REAL fn1, REAL fs1, REAL fw1, REAL fe1)
{
    int FILTER_HEIGHT = 5;
    int FILTER_WIDTH = 5;
    const int PROCESS_DATA_COUNT = 4;
    const int DATA_CACHE_SIZE = PROCESS_DATA_COUNT + FILTER_HEIGHT - 1;
    /*
    REAL filter[FILTER_HEIGHT][FILTER_WIDTH] = {
        { 0,  0, 1, 0, 0, },
        { 0,  0, 2, 0, 0, },
        { 3,  4, 5, 6, 7, },
        { 0,  0, 8, 0, 0, },
        { 0,  0, 9, 0, 0, },
    };
    int flt_width = 5;
    int flt_height = 5;
    */
    int i, j, k;
#pragma omp parallel for num_threads(2)
    for (i = 0; i < height; i++)
    {
#pragma omp parallel for num_threads(4)
        for (j = 0; j < width; j += 1)
        {
            //REAL sum[PROCESS_DATA_COUNT];
            REAL sum[DATA_CACHE_SIZE];
//#pragma omp parallel for num_threads(8) ordered
            //for (k = 0; k < DATA_CACHE_SIZE; k++)
            //{
                //REAL sum1 = 0;
                /*
        for (int n = 0; n < flt_width; n++) {
            for (int m = 0; m < flt_height; m++) {
                int x = j + n - flt_width / 2;
                int y = i + m - flt_height / 2;
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    //buf[m*flt_width + n] = src[y*width + x];
                    int idx = m*flt_width + n;
                    sum1 += src[y*width + x] * filter[m][n];
                }
            }
        }*/
                
                int p, idx_x, idx_y;
                REAL data[DATA_CACHE_SIZE + 1];
                int tidy = i;
                int tidx = j;
                int index = tidy * width + tidx;
                /*
                if (tidx < 0)               index -= tidx;
                else if (tidx >= width)     index -= tidx - width + 1;
                if (tidy < 0)               index -= tidy*width;
                else if (tidy >= height)    index -= (tidy - height + 1)*width;
                for (p = 0; p < FILTER_HEIGHT; p++)
                {
                    int _tidy = tidy + p;
                    //printf("get src: %d, %d --- %d\n", tidy, tidx, index);
                    //printf("array size: %d\n", sizeof(*src));
                    data[p] = src[index];
                    //printf("done get src\n");
                    if (_tidy >= 0 && _tidy < height - 1) {
                        index += width;
                    };
                };
                */
                data[2] = src[index];
                // read vertical
                if (tidy - 2 < 0) data[0] = src[tidx];
                else data[0] = src[index - 2*width];
                if (tidy - 1 < 0) data[1] = src[tidx];
                else data[1] = src[index - width];
                if (tidy + 1 >= height) data[3] = src[index];
                else data[3] = src[index + width];
                if (tidy + 2 >= height) data[4] = src[index];
                else data[4] = src[index + 2*width];

                // read horizontal
                if (tidx - 1 < 0) data[5] = src[index];
                else data[5] = src[index - 1];
                if (tidx - 2 < 0) data[6] = src[index];
                else data[6] = src[index - 2];
                if (tidx + 1 >= width) data[7] = src[index];
                else data[7] = src[index + 1];
                if (tidx + 2 >= width) data[8] = src[index];
                else data[8] = src[index + 2];
                //int tid = omp_get_thread_num();
                int tid = 0;
                sum[tid] = data[8] * fe1;
                // shuffle
//#pragma omp ordered threads
//                   if (tid < 8)
//                   {
//                       sum[tid] = sum[tid + 1];
//                   };
                sum[tid] += data[7] * fe0;
//#pragma omp ordered threads
//                   if (tid < 8)
//                  {
//                     sum[tid] = sum[tid + 1];
//                 };
                sum[tid] += data[0] * fn1;
                sum[tid] += data[1] * fn0;
                sum[tid] += data[2] * fc;
                sum[tid] += data[3] * fs0;
                sum[tid] += data[4] * fs1;
//#pragma omp ordered threads
//                    if (tid < 8)
//                    {
//                        sum[tid] = sum[tid + 1];
//                    };
                sum[tid] += data[5] * fw0;
                sum[tid] += data[6] * fw1;
/*#pragma omp ordered threads
                if (tid < 8)
                {
                    sum[tid] = sum[tid + 1];
                };
                sum[tid] += data[6] * fw1;
            };
            for (k = 0; k < 4; k++)
            {
                //printf("save to dst\n");
                dst[i * width + j + k] = sum[k];
                //printf("done save to dst\n");
            };*/
        dst[i*width + j] = sum[0];
        };
    };
}

bool prepareData (REAL* data, int w, int h) {
    char szPath[1024] = "";
	sprintf(szPath, "../Lena_%dx%d.raw", w, h);
	bool bRtn = false;
	FILE* fp = fopen(szPath, "rb");
	if (fp) {
		if (w*h == fread(data, sizeof(REAL), w*h, fp)) {
			bRtn = true;
		}
		fclose(fp);
	}
    return bRtn;
}

int hub(int width, int height) {
    REAL* input = (REAL*)malloc(width*height*sizeof(REAL));
    REAL* output = (REAL*)malloc(width*height*sizeof(REAL));
    prepareData(input, width, height);

    // prepare filter matrix
    int FILTER_HEIGHT = 5;
    int FILTER_WIDTH = 5;
    REAL filter[FILTER_HEIGHT][FILTER_WIDTH] = {
        { 0,  0, 1, 0, 0, },
        { 0,  0, 2, 0, 0, },
        { 3,  4, 5, 6, 7, },
        { 0,  0, 8, 0, 0, },
        { 0,  0, 9, 0, 0, },
    };

    REAL fc = filter[2][2];
    REAL fn0 = filter[1][2];
    REAL fs0 = filter[3][2];
    REAL fw0 = filter[2][1];
    REAL fe0 = filter[2][3];
    REAL fn1 = filter[0][2];
    REAL fs1 = filter[4][2];
    REAL fw1 = filter[2][0];
    REAL fe1 = filter[2][4];

    // call the CPU kernel
    j2d9pt_cpu(input, output, width, height, fc, fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1);

    // prepare the data for verification
    REAL* verify_data = (REAL*)malloc(width*height*sizeof(REAL));
    Convolution(input, verify_data, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);

    double dif = 0;
    for (int i = 0; i < width*height; i++) {
        int x = i % width;
        int y = i / width;
        if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
            dif += abs(verify_data[i] - output[i]);
    }
    printf("verify dif =%f\n", dif);
    return 0;
}
