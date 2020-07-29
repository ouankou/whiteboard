#include "common.h"
#include "cudaLib.cuh"
#include <omp.h>
#define REAL double

namespace stencil2d_9pt {
	static const int WARP_SIZE = 32;
	static const int FILTER_WIDTH = 5;
	static const int FILTER_HEIGHT = 5;

	void j2d9pt(const REAL* __restrict__ src, REAL* dst, int width, int height, REAL fc, REAL fn0, REAL fs0, REAL fw0, REAL fe0, REAL fn1, REAL fs1, REAL fw1, REAL fe1)
	{
        // 2 gird, 4 block, multiple mini warps
        // mini warp: 8 threads: 4 for data points, 4 as helper
		const int PROCESS_DATA_COUNT = 4;
		const int DATA_CACHE_SIZE = PROCESS_DATA_COUNT + FILTER_HEIGHT - 1;
        REAL filter[FILTER_HEIGHT][FILTER_WIDTH] = {
			{ 0,  0, 1, 0, 0, },
			{ 0,  0, 2, 0, 0, },
			{ 3,  4, 5, 6, 7, },
			{ 0,  0, 8, 0, 0, },
			{ 0,  0, 9, 0, 0, },
		};
        int flt_width = 5;
        int flt_height = 5;
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
                    REAL sum1 = 0;
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

	template<class DataType, int PROCESS_DATA_COUNT, int BLOCK_SIZE>
	static void Test(int width, int height) {
		const int WARP_COUNT = BLOCK_SIZE >> 5;
		const int WARP_PROCESS_DATA_COUNT = WARP_SIZE - FILTER_WIDTH + 1;
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT*WARP_COUNT;

		const int nRepeatCount = 1;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		//StopWatchWin watch;
		DataT<DataType> img;
		char szPath[1024] = "";
		sprintf(szPath, "../Lena_%dx%d.raw", width, height);
		bool bRtn = img.Load_uchar(szPath, width, height);
		//sprintf(szPath, "../data/Lena%dx%d.txt", width, height);
		//img.SaveText(szPath);
		if (!bRtn) {
			printf("Load failed : %s, generate random data\n", szPath);
			img.MallocBuffer(width, height);
			for (int i = 0; i < img.width*img.height; i++) {
				img.data[i] = std::rand() % 256;
				//img.data[i] = i/img.width;
			}
		}
		else {
			printf("Load success : %s\n", szPath);
		}
		//sprintf(szPath, "../Lena_%dx%d.raw", width, height);
		//img.SaveRaw(szPath);

		DevBuffer<DataType> devSrc(width, height), devDst(width, height);
		devSrc.CopyFromHost(img.data, img.width, img.width, img.height);
		DataT<DataType> imgDst;
		imgDst.MallocBuffer(width, height);

		//dim3 block_size(BLOCK_SIZE, 1);
		dim3 grid_size(UpDivide(width, BLOCK_PROCESS_DATA_COUNT), UpDivide(height, PROCESS_DATA_COUNT));

		DataType filter[FILTER_HEIGHT][FILTER_WIDTH] = {
			{ 0,  0, 1, 0, 0, },
			{ 0,  0, 2, 0, 0, },
			{ 3,  4, 5, 6, 7, },
			{ 0,  0, 8, 0, 0, },
			{ 0,  0, 9, 0, 0, },
		};

		DataType fc = filter[2][2];
		DataType fn0 = filter[1][2];
		DataType fs0 = filter[3][2];
		DataType fw0 = filter[2][1];
		DataType fe0 = filter[2][3];
		DataType fn1 = filter[0][2];
		DataType fs1 = filter[4][2];
		DataType fw1 = filter[2][0];
		DataType fe1 = filter[2][4];

        //printf("src: %f\n", ((double*)(devSrc.GetData()))[0]);
		cudaEventRecord(start, 0);
		for (int s = 0; s < nRepeatCount; s++) {
			//j2d9pt(devSrc.GetData(), devDst.GetData(), width, height, fc, fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1);
			j2d9pt(img.data, imgDst.data, width, height, fc, fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1);
		}
		cudaDeviceSynchronize();
		//watch.stop();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		CUDA_CHECK_ERROR;

		//devDst.CopyToHost(imgDst.data, imgDst.width, imgDst.width, imgDst.height);

		cudaEventElapsedTime(&inc, start, stop);
		//inc = watch.getAverageTime();
		inc /= (float)nRepeatCount;
		printf("%dx%d , %dx%d , proc_count=%d, cache=%d, BLOCK_SIZE=%d, %f ms , %f fps\n", width, height, FILTER_WIDTH, FILTER_HEIGHT, PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, inc, 1000.0 / inc);
		sprintf(szPath, "../Lena_omp_proc_%dx%d.raw", width, height);
		imgDst.SaveRaw(szPath);

		sprintf(szPath, "../Lena_omp_proc_%dx%d.txt", width, height);
		imgDst.SaveText(szPath);

		DataT<DataType> imgVerify;
		imgVerify.MallocBuffer(width, height);
		Convolution(img.data, imgVerify.data, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
		sprintf(szPath, "../data/Lena_proc_verify_%dx%d.txt", width, height);
		//imgVerify.SaveText(szPath);

		double dif = 0;
		for (int i = 0; i < img.width*img.height; i++) {
			int x = i % img.width;
			int y = i / img.width;
			if (x > FILTER_WIDTH/2 && x < width - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height - FILTER_HEIGHT/2)
				dif += abs(imgVerify.data[i] - imgDst.data[i]);
		}
		printf("verify dif =%f\n", dif);
		sprintf(szPath, "../data/Lena_proc_verify_%dx%d.txt", width, height);
		//imgVerify.SaveText(szPath);
		sprintf(szPath, "../data/Lena_proc_verify(%dx%d)_%dx%d.raw", FILTER_WIDTH, FILTER_HEIGHT, width, height);
		//imgVerify.SaveRaw(szPath);
	}
};

template<typename T>
static int stencil_9pt(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_9pt::Test<T, P, B>(size, size);
	return 0;
}

int stencil_9pt_double(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_9pt::Test<double, P, B>(size, size);
	return 0;
}
/*
int stencil_9pt_float(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_9pt::Test<float, P, B>(size, size);
	return 0;
}
*/


