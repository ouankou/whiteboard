#include "common.h"
#include "cudaLib.cuh"

namespace stencil2d_9pt {
	static const int WARP_SIZE = 32;
	static const int FILTER_WIDTH = 5;
	static const int FILTER_HEIGHT = 5;

	template<typename T, int BLOCK_SIZE, int PROCESS_DATA_COUNT>
	__global__ void j2d9pt(const T* __restrict__ src, T* dst, int width, int height, 
		T fc, T fn0, T fs0, T fw0, T fe0, T fn1, T fs1, T fw1, T fe1)
	{
		const int WARP_COUNT = BLOCK_SIZE >> 5;
		const int laneId = threadIdx.x & 31;
		const int warpId = threadIdx.x >> 5;
		const int WARP_PROCESS_DATA_COUNT = WARP_SIZE - FILTER_WIDTH + 1;
		const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT*WARP_COUNT;
		const int DATA_CACHE_SIZE = PROCESS_DATA_COUNT + FILTER_HEIGHT - 1;

		T data[DATA_CACHE_SIZE];

		const int process_count = BLOCK_PROCESS_DATA_COUNT*blockIdx.x + WARP_PROCESS_DATA_COUNT*warpId;
		if (process_count >= width)
			return;
		int tidx = process_count + laneId - FILTER_WIDTH / 2;
		int tidy = PROCESS_DATA_COUNT*blockIdx.y - FILTER_HEIGHT / 2;

		{
			int index = width*tidy + tidx;
			if (tidx < 0)            index -= tidx;
			else if (tidx >= width)  index -= tidx - width + 1;
			if (tidy < 0)            index -= tidy*width;
			else if (tidy >= height) index -= (tidy - height + 1)*width;

#pragma unroll
			for (int s = 0; s < DATA_CACHE_SIZE; s++) {
				int _tidy = tidy + s;
				data[s] = src[index];
				if (_tidy >= 0 && _tidy < height - 1) {
					//data[s] = src[index];
					index += width;
				}
				//else {
				//	data[s] = 0;
				//}
			}
		}
	
		#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			T sum = data[i + 2] * fe1;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 2] * fe0;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 0] * fn1;
			sum += data[i + 1] * fn0;
			sum += data[i + 2] * fc;
			sum += data[i + 3] * fs0;
			sum += data[i + 4] * fs1;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 2] * fw0;

			sum = __my_shfl_down(sum, 1);
			sum += data[i + 2] * fw1;

			data[i] = sum;
		}

		if (laneId >= WARP_SIZE - FILTER_WIDTH + 1)
			return;

		int _x = tidx + FILTER_WIDTH / 2;
		int _y = tidy + FILTER_HEIGHT / 2;
		int index = width*_y + _x;
		if (_x >= width)
			return;
#pragma unroll
		for (int i = 0; i < PROCESS_DATA_COUNT; i++) {
			if (_y + i < height) {
				dst[index] = data[i];
				index += width;
			}
		}
	}

	template<class DataType, int PROCESS_DATA_COUNT, int BLOCK_SIZE>
	static float Test(int width, int height) {
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
		sprintf(szPath, "../data/Lena%dx%d.raw", width, height);
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

		DevBuffer<DataType> devSrc(width, height), devDst(width, height);
		devSrc.CopyFromHost(img.data, img.width, img.width, img.height);
		DataT<DataType> imgDst;
		imgDst.MallocBuffer(width, height);

		dim3 block_size(BLOCK_SIZE, 1);
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

		cudaEventRecord(start, 0);
		for (int s = 0; s < nRepeatCount; s++) {
			j2d9pt<DataType, BLOCK_SIZE, PROCESS_DATA_COUNT> <<<grid_size, block_size >>> 
				(devSrc.GetData(), devDst.GetData(), width, height, fc, fn0, fs0, fw0, fe0, fn1, fs1, fw1, fe1);
		}
		cudaDeviceSynchronize();
		//watch.stop();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		CUDA_CHECK_ERROR;

		devDst.CopyToHost(imgDst.data, imgDst.width, imgDst.width, imgDst.height);

		cudaEventElapsedTime(&inc, start, stop);
		//inc = watch.getAverageTime();
		inc /= (float)nRepeatCount;
		printf("%dx%d , %dx%d , proc_count=%d, cache=%d, BLOCK_SIZE=%d, %f ms , %f fps\n", width, height, FILTER_WIDTH, FILTER_HEIGHT, PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, inc, 1000.0 / inc);
		sprintf(szPath, "../data/Lena_proc_%dx%d.raw", width, height);
		//imgDst.SaveRaw(szPath);

		sprintf(szPath, "../data/Lena_proc_%dx%d.txt", width, height);
		//imgDst.SaveText(szPath);

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
#if 0
		FILE* fp = fopen("log.conv2D.csv", "at");
		if (fp) {
			fprintf(fp, "%dx%d, %d_%d, %d, %dx%d, %f\n", width, height, PROCESS_DATA_COUNT, PROCESS_DATA_COUNT + FILTER_HEIGHT - 1, BLOCK_SIZE, FILTER_WIDTH, FILTER_HEIGHT, inc);
			fclose(fp);
		}
		return inc;
#endif
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
int stencil_9pt_float(int argc, char** argv) {
	DISPLAY_FUNCTION("");
	printf("datatype=double\n");
	int size = 8192; if (argc > 1) size = atoi(argv[1]);
	const int P = 4;
	const int B = 128;
	stencil2d_9pt::Test<float, P, B>(size, size);
	return 0;
}


