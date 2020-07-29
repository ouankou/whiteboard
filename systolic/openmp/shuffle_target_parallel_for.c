#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>

#define REAL double

void Convolution(const REAL* src, REAL* dst, int width, int height, const REAL* filter, int flt_width, int flt_height)
{
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

int main (int argc, char** argv) {

    int width = 4096;
    int height = 4096;
    REAL* input = (REAL*)malloc(width*height*sizeof(REAL));
    REAL* output = (REAL*)malloc(width*height*sizeof(REAL));
    prepareData(input, width, height);

    // prepare filter matrix
    int FILTER_HEIGHT = 5;
    int FILTER_WIDTH = 5;
    REAL filter[5][5] = {
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
    const int PROCESS_DATA_COUNT = 4;
    const int DATA_CACHE_SIZE = PROCESS_DATA_COUNT + FILTER_HEIGHT - 1;
    REAL* src = input;
    REAL* dst = output;
    int i;
    omp_set_nested(1);
//#pragma omp target map(to: src[0:width*height], fc, fn0, fn1, fw1, fw0, fe1, fe0, fs1, fs0, height, width) map(from: dst[0:width*height])
#pragma omp parallel for num_threads(2)
    for (i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j += 4)
        {
            //REAL sum[PROCESS_DATA_COUNT];
            REAL __shuffle_buffer[8] = {11, 22, 33, 44, 55, 66, 77, 88};
            //omp_set_nested(1);
            int turn = 0;
#pragma omp parallel num_threads(8) shared(__shuffle_buffer, turn)
            {
                int sa_id = omp_get_thread_num();
                //printf("ID %d in PARALLEL (i=%d, j=%d): %lf\n", sa_id, i, j, __shuffle_buffer[sa_id]);
                //REAL sum[8];
                REAL data[5];
                int tidy = i - 2;
                int tidx = j + sa_id - 2;
                int index = tidy * width + tidx;

                if (tidx < 0)            index -= tidx;
                else if (tidx >= width)  index -= tidx - width + 1;
                if (tidy < 0)            index -= tidy*width;
                else if (tidy >= height) index -= (tidy - height + 1)*width;

                for (int s = 0; s < 5; s++) {
                    int _tidy = tidy + s;
                    data[s] = src[index];
                    if (_tidy >= 0 && _tidy < height - 1) {
                        index += width;
                    }
                }
                /*
                // read horizontal
                if (tidx < 0) index -= tidx;
                else data[5] = src[index - 1];
                if (tidx - 2 < 0) data[6] = src[index];
                else data[6] = src[index - 2];
                if (tidx + 1 >= width) data[7] = src[index];
                else data[7] = src[index + 1];
                if (tidx + 2 >= width) data[8] = src[index];
                else data[8] = src[index + 2];

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
                */

                for (int t = 0; t < 1; t++) {
                    __shuffle_buffer[sa_id] = data[t + 2] * fe1;

                    //sum = __my_shfl_down(sum, 1);
                    #pragma omp barrier
                    while (sa_id != turn) { ; }
                    if (sa_id < 7)
                    __shuffle_buffer[sa_id] = __shuffle_buffer[sa_id + 1];
                    turn = sa_id + 1;
                    if (sa_id == omp_get_num_threads() - 1) turn = 0;
                    #pragma omp barrier
                    __shuffle_buffer[sa_id] += data[t + 2] * fe0;

                    //sum = __my_shfl_down(sum, 1);
                    #pragma omp barrier
                    while (sa_id != turn) { ; }
                    if (sa_id < 7)
                    __shuffle_buffer[sa_id] = __shuffle_buffer[sa_id + 1];
                    turn = sa_id + 1;
                    if (sa_id == omp_get_num_threads() - 1) turn = 0;
                    #pragma omp barrier
                    __shuffle_buffer[sa_id] += data[t + 0] * fn1;
                    __shuffle_buffer[sa_id] += data[t + 1] * fn0;
                    __shuffle_buffer[sa_id] += data[t + 2] * fc;
                    __shuffle_buffer[sa_id] += data[t + 3] * fs0;
                    __shuffle_buffer[sa_id] += data[t + 4] * fs1;

                    //sum = __my_shfl_down(sum, 1);
                    #pragma omp barrier
                    while (sa_id != turn) { ; }
                    if (sa_id < 7)
                    __shuffle_buffer[sa_id] = __shuffle_buffer[sa_id + 1];
                    turn = sa_id + 1;
                    if (sa_id == omp_get_num_threads() - 1) turn = 0;
                    #pragma omp barrier
                    __shuffle_buffer[sa_id] += data[t + 2] * fw0;

                    //sum = __my_shfl_down(sum, 1);
                    #pragma omp barrier
                    while (sa_id != turn) { ; }
                    if (sa_id < 7)
                    __shuffle_buffer[sa_id] = __shuffle_buffer[sa_id + 1];
                    turn = sa_id + 1;
                    if (sa_id == omp_get_num_threads() - 1) turn = 0;
                    #pragma omp barrier
                    __shuffle_buffer[sa_id] += data[t + 2] * fw1;

                    data[t] = __shuffle_buffer[sa_id];
                }
                /*
                int tid = 0;
                sum[tid] = data[8] * fe1;
                sum[tid] += data[7] * fe0;
                sum[tid] += data[0] * fn1;
                sum[tid] += data[1] * fn0;
                sum[tid] += data[2] * fc;
                sum[tid] += data[3] * fs0;
                sum[tid] += data[4] * fs1;
                sum[tid] += data[5] * fw0;
                sum[tid] += data[6] * fw1;
                dst[i*width + j] = sum[0];
                */

                if (sa_id < 4) {
                //    return;

                int _x = tidx + 2;
                int _y = tidy + 2;
                index = width * _y + _x;
                if (_x < width)
                    for (int t = 0; t < 1; t++) {
                        if (_y + t < height) {
                            dst[index] = data[t];
                            index += width;
                        }
                    }}
            };
        };
    };

    // prepare the data for verification
    REAL* verify_data = (REAL*)malloc(width*height*sizeof(REAL));
    Convolution(input, verify_data, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);

    double dif = 0;
    for (int i = 0; i < width*height; i++) {
        int x = i % width;
        int y = i / width;
        if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
            dif += fabs(verify_data[i] - output[i]);
    }
    printf("verify dif =%f\n", dif);

    return 0;
}
