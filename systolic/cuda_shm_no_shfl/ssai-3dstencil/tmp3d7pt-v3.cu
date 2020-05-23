#include <stdio.h>
#include <cassert>
#include <cstdio>
#include "common.hpp"

namespace v3 {
	////////////////////////////////////
#if 1
#define    _F3d7PT_011         ((T)0.100000)
#define    _F3d7PT_101         ((T)0.200000)
#define    _F3d7PT_110         ((T)0.500000)
#define    _F3d7PT_111         ((T)0.300000)
#define    _F3d7PT_112         ((T)0.600000)
#define    _F3d7PT_121         ((T)0.400000)
#define    _F3d7PT_211         ((T)0.700000)
#else
#pragma message("using debug parameters")
#define    _F3d7PT_011         0
#define    _F3d7PT_101         0
#define    _F3d7PT_110         0
#define    _F3d7PT_111         0
#define    _F3d7PT_112         0
#define    _F3d7PT_121         0
#define    _F3d7PT_211         1
#endif
////////////////////////////////////
#define _FILTER_SIZE  3
	const int FILTER_WIDTH = _FILTER_SIZE;
	const int FILTER_HEIGHT = _FILTER_SIZE;
	const int FILTER_DEPTH = _FILTER_SIZE;
	////////////////////////////////////

	//typedef double REAL;
	static const int WARP_SIZE = 32;

#define max(x,y)  ((x) > (y)? (x) : (y))
#define min(x,y)  ((x) < (y)? (x) : (y))
#define ceil(a,b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1)

	__device__ __host__ __forceinline__ unsigned int UpDivide(unsigned int x, unsigned int y) { assert(y != 0); return (x + y - 1) / y; }
	__device__ __host__ __forceinline__ unsigned int UpRound(unsigned int x, unsigned int y) { return UpDivide(x, y)*y; }
#if (CUDART_VERSION >= 9000)
#pragma message("CUDART_VERSION >= 9000")
#define __my_shfl_up(var, delta) __shfl_up_sync(0xFFFFFFFF, var, delta)
#define __my_shfl_down(var, delta) __shfl_down_sync(0xFFFFFFFF, var, delta)
#else
#pragma message("CUDART_VERSION < 9000")
#define __my_shfl_up(var, delta) __shfl_up(var, delta)
#define __my_shfl_down(var, delta) __shfl_down(var, delta)
#endif

#define MAD(__x, __y, __z) ((__x)*(__y)+(__z))

	static void check_error(const char* message) {
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("CUDA error : %s, %s\n", message, cudaGetErrorString(error));
			exit(-1);
		}
	}

	template<typename T, int FILTER_SIZE, int PROCESS_DATA_COUNT_Y, int PROCESS_DATA_COUNT_Z, int WARP_COUNT, int ITERATIVE_COUNT>
	__global__ void kernel3d_3d7pt(const T * __restrict__ src, T* dst,
		int nx, int ny, int nz, int nxy) {
		//assert(ITERATIVE_COUNT == 1);
		const int FILTER_WIDTH = FILTER_SIZE;
		const int FILTER_HEIGHT = FILTER_SIZE;
		const int FILTER_DEPTH = FILTER_SIZE;
		const int HALF_FILTER_WIDTH = FILTER_WIDTH / 2;
		const int HALF_FILTER_HEIGHT = FILTER_HEIGHT / 2;
		const int HALF_FILTER_DEPTH = FILTER_DEPTH / 2;
#ifdef _DEBUG
		const int laneId = threadIdx.x;
		const int warpId = threadIdx.y;
#else
#define laneId  (threadIdx.x)
#define warpId  (threadIdx.y)
#endif
		const int WARP_PROCESS_DATA_COUNT_X = WARP_SIZE - (FILTER_WIDTH - 1) * ITERATIVE_COUNT;
		const int DATA_CACHE_SIZE = PROCESS_DATA_COUNT_Y + (FILTER_HEIGHT - 1) * ITERATIVE_COUNT;
		assert(WARP_COUNT == PROCESS_DATA_COUNT_Z + (FILTER_SIZE - 1) * ITERATIVE_COUNT);

		const int SMEM_CACHE_SIZE = (DATA_CACHE_SIZE - (FILTER_HEIGHT - 1)) / 2;
		//__shared__ T sMem[WARP_COUNT - HALF_FILTER_DEPTH * 2][DATA_CACHE_SIZE - HALF_FILTER_HEIGHT * 2][WARP_SIZE - HALF_FILTER_WIDTH * 2];
		__shared__ T sMem[WARP_COUNT - HALF_FILTER_DEPTH * 2][SMEM_CACHE_SIZE][WARP_SIZE - HALF_FILTER_WIDTH * 2];

		T data[DATA_CACHE_SIZE];

#ifdef _DEBUG
		const int width = nx;
		const int height = ny;
		const int depth = nz;
#else
#define width nx
#define height ny
#define depth  nz
#endif
		const int process_count_x = WARP_PROCESS_DATA_COUNT_X*blockIdx.x;
		const int process_count_y = PROCESS_DATA_COUNT_Y*blockIdx.y;
		const int process_count_z = PROCESS_DATA_COUNT_Z*blockIdx.z;

		const int tidx = process_count_x + laneId - HALF_FILTER_WIDTH * ITERATIVE_COUNT;
		const int tidy = process_count_y - HALF_FILTER_HEIGHT * ITERATIVE_COUNT;
		const int tidz = process_count_z + warpId - HALF_FILTER_DEPTH * ITERATIVE_COUNT;

		{
			int index = nxy*tidz + nx*tidy + tidx;
			if (tidx < 0)        index -= tidx;
			else if (tidx >= nx) index -= tidx - nx + 1;
			if (tidz < 0)        index -= tidz*nxy;
			else if (tidz >= nz) index -= (tidz - nz + 1)*nxy;
			if (tidy < 0)        index -= tidy*nx;
			else if (tidy >= ny) index -= (tidy - ny + 1)*nx;

#pragma unroll
			for (int s = 0; s < DATA_CACHE_SIZE; s++) {
				int _tidy = tidy + s;
				data[s] = src[index];
				if (_tidy >= 0 && _tidy < ny - 1) index += nx;
			}
		}

		//iterative compute
#pragma unroll
		for (int ite = 0; ite < ITERATIVE_COUNT; ite++) {
			const int CURRENT_PROCESS_DATA = DATA_CACHE_SIZE - (FILTER_HEIGHT - 1)*(ite + 1);
			if (warpId >= HALF_FILTER_DEPTH*ite && warpId < WARP_COUNT - HALF_FILTER_DEPTH *ite)
			{
#pragma unroll
				for (int i = 0; i < CURRENT_PROCESS_DATA; i++) {
					if (laneId >= ite + 1 && laneId < WARP_SIZE - ite - 1) {
						if (warpId >= 0 + ite && warpId < WARP_COUNT - HALF_FILTER_DEPTH * 2 - ite) {
							T sum = data[i + 1] * _F3d7PT_011;
							//sMem[warpId + 1 - HALF_FILTER_DEPTH][i + HALF_FILTER_HEIGHT - HALF_FILTER_HEIGHT][laneId - HALF_FILTER_WIDTH] = sum;
							sMem[warpId + 1 - HALF_FILTER_DEPTH][0][laneId - HALF_FILTER_WIDTH] = sum;
						}
					}
					__syncthreads();

					if (laneId >= ite + 1 && laneId < WARP_SIZE - ite - 1) {
						if (warpId >= HALF_FILTER_DEPTH * 2 + ite && warpId < WARP_COUNT - ite) {
							T sum = data[i + 1] * _F3d7PT_211;
							//sMem[warpId - 1 - HALF_FILTER_DEPTH][i + HALF_FILTER_HEIGHT - HALF_FILTER_HEIGHT][laneId - HALF_FILTER_WIDTH] += sum;
							sMem[warpId - 1 - HALF_FILTER_DEPTH][0][laneId - HALF_FILTER_WIDTH] += sum;
						}
					}
					__syncthreads();

					if (warpId >= 0 + ite + 1 && warpId < WARP_COUNT - ite - 1) {
						T sum = 0;
						{
							//m = 0
							//sum = MAD(data[i + 0], _F000, sum);
							sum = MAD(data[i + 1], _F3d7PT_110, sum);
							//sum = MAD(data[i + 2], _F020, sum);
						}
						{
							//m = 2
							sum = __my_shfl_up(sum, 2);
							//sum = MAD(data[i + 0], _F002, sum);
							sum = MAD(data[i + 1], _F3d7PT_112, sum);
							//sum = MAD(data[i + 2], _F022, sum);
						}
						{
							//m = 1
							sum = __my_shfl_down(sum, 1);
							sum = MAD(data[i + 0], _F3d7PT_101, sum);
							sum = MAD(data[i + 1], _F3d7PT_111, sum);
							sum = MAD(data[i + 2], _F3d7PT_121, sum);
						}
						if (laneId >= HALF_FILTER_WIDTH*(ite + 1) && laneId < WARP_SIZE - HALF_FILTER_WIDTH*(ite + 1)) {
							//data[i] = sum + sMem[warpId - HALF_FILTER_DEPTH][i + HALF_FILTER_HEIGHT - HALF_FILTER_HEIGHT][laneId - HALF_FILTER_WIDTH];
							data[i] = sum + sMem[warpId - HALF_FILTER_DEPTH][0][laneId - HALF_FILTER_WIDTH];
						}
					}
					__syncthreads();
				}
			}
			else {
				return;
			}
			__syncthreads();
		}

		if (warpId < HALF_FILTER_DEPTH * ITERATIVE_COUNT || warpId >= WARP_COUNT - HALF_FILTER_DEPTH * ITERATIVE_COUNT) {
			return;
		}
		if (laneId < HALF_FILTER_WIDTH * ITERATIVE_COUNT || laneId >= WARP_SIZE - HALF_FILTER_WIDTH * ITERATIVE_COUNT)
			return;

		//save to gmem
		{
#ifdef _DEBUG
			const int& _tidx = tidx;
			const int& _tidz = tidz;
#else
#define _tidx tidx
#define _tidz tidz
#endif
			int _tidy = tidy + HALF_FILTER_HEIGHT * ITERATIVE_COUNT;

			//int _tidx = tidx - (FILTER_WIDTH - 1) / 2*2;
			int index = _tidz*nxy + nx*_tidy + _tidx;
#pragma unroll
			for (int i = 0; i < PROCESS_DATA_COUNT_Y; i++) {
				{
					assert(_tidx >= 0);
					assert(_tidy >= 0);
					assert(_tidz >= 0);
					if (_tidx < nx && _tidy < ny && _tidz < nz) {
						//dst[index] = sMem[warpId][i][laneId];
						dst[index] = data[i];
					}
				}
				_tidy++;
				index += nx;
			}
		}
#ifdef _DEBUG
#else
#undef laneId
#undef warpId
#undef width
#undef height
#undef depth
#endif
	}



	template<typename T, int ITERATIVE_COUNT, int BLOCK_SIZE, int CACHE_DATA_COUNT>
	static void host_code(T *h_in, T *h_out, int N) {
		T *in;
		cudaMalloc(&in, sizeof(T)*N*N*N);
		check_error("Failed to allocate device memory for in\n");
		cudaMemcpy(in, h_in, sizeof(T)*N*N*N, cudaMemcpyHostToDevice);
		T *out;
		cudaMalloc(&out, sizeof(T)*N*N*N);
		check_error("Failed to allocate device memory for out\n");

		{
			//V100, It 3, 14 24
#if 0
			const int CACHE_DATA_COUNT = 16;
			const int WARP_COUNT = 24;
#else
		//const int CACHE_DATA_COUNT = 12;
		//const int WARP_COUNT = 16;
			const int WARP_COUNT = BLOCK_SIZE / WARP_SIZE;
#endif
			const int PROCESS_DATA_COUNT = CACHE_DATA_COUNT - (FILTER_WIDTH - 1)*ITERATIVE_COUNT;
			const int BLOCK_PROCESS_DATA_COUNT_X = WARP_SIZE - (FILTER_WIDTH - 1)*ITERATIVE_COUNT;
			const int BLOCK_PROCESS_DATA_COUNT_Y = PROCESS_DATA_COUNT;
			const int BLOCK_PROCESS_DATA_COUNT_Z = WARP_COUNT - (FILTER_DEPTH - 1)*ITERATIVE_COUNT;

			{
				assert(WARP_COUNT >= FILTER_DEPTH);

				int flag = 0;
				const int nx_ = N;
				const int ny_ = N;
				const int nz_ = N;
				size_t s = sizeof(T) * nx_ * ny_ * nz_;
				dim3 block_size(WARP_SIZE, WARP_COUNT);
				dim3 grid_size(UpDivide(nx_, BLOCK_PROCESS_DATA_COUNT_X),
					UpDivide(ny_, BLOCK_PROCESS_DATA_COUNT_Y),
					UpDivide(nz_, BLOCK_PROCESS_DATA_COUNT_Z)
				);
				float ftime = 0;
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);

				cudaEventRecord(start, 0);
				kernel3d_3d7pt<T, _FILTER_SIZE, BLOCK_PROCESS_DATA_COUNT_Y, BLOCK_PROCESS_DATA_COUNT_Z, WARP_COUNT, ITERATIVE_COUNT> << <grid_size, block_size >> >
					(in, out, nx_, ny_, nz_, nx_*ny_);

				cudaDeviceSynchronize();
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&ftime, start, stop);

				cudaMemcpy(h_out, out, sizeof(T)*N*N*N, cudaMemcpyDeviceToHost);
				check_error("Failed to copy device memory to host\n");
				double gups = double(N)*N*N*0.000001 / ftime*ITERATIVE_COUNT;
				double flops = gups * 13;

				printf("dtype=%s, all_time=%fms, one_itr_time=%fms, N=%d,Iterative=%d, CacheCount=%d, WarpCount=%d,  gups=%.2fGUPS, gflop=%.2fGFLOP/s\n",
					sizeof(T) == 8 ? "double" : "float", ftime, ftime / ITERATIVE_COUNT, N, ITERATIVE_COUNT, CACHE_DATA_COUNT, WARP_COUNT, gups, flops);
			}
		}

		cudaFree(in);
		cudaFree(out);
	}

	template<typename T, int _N> static
		void j3d7pt_gold(const T* l_in, T* l_out, int N) {
		const T(*in)[_N][_N] = (const T(*)[_N][_N])l_in;
		T(*out)[_N][_N] = (T(*)[_N][_N])l_out;

#pragma omp parallel for
		for (int k = 1; k < N - 1; k++) {
			for (int j = 1; j < N - 1; j++) {
				for (int i = 1; i < N - 1; i++) {
					out[k][j][i] =
						_F3d7PT_011 * in[k - 1][j][i] +
						_F3d7PT_101 * in[k][j - 1][i] +
						_F3d7PT_111 * in[k][j][i] +
						_F3d7PT_121 * in[k][j + 1][i] +
						_F3d7PT_110 * in[k][j][i - 1] +
						_F3d7PT_112 * in[k][j][i + 1] +
						_F3d7PT_211 * in[k + 1][j][i];
				}
			}
		}
	}

	template<typename T, int ITERATIVE_COUNT, int N, int BLOCK_SIZE, int CACHE_DATA_COUNT>
	static int _3d7pt(int argc, char** argv) {
		printf("%s:%s:Datatype:%d\n", __FILE__, __FUNCTION__, sizeof(T));
		double error = 0;

		T(*input)[N][N] = (T(*)[N][N]) getRandom3DArray<T>(N, N, N);
		T(*output)[N][N] = (T(*)[N][N]) getZero3DArray<T>(N, N, N);
		T(*output_gold)[N][N] = (T(*)[N][N]) getZero3DArray<T>(N, N, N);

		//const int ITERATIVE_COUNT = 3;
		host_code<T, ITERATIVE_COUNT, BLOCK_SIZE, CACHE_DATA_COUNT>((T*)input, (T*)output, N);

		for (int i = 0; i < ITERATIVE_COUNT; i++) {
			j3d7pt_gold<T, N>((T*)input, (T*)output_gold, N);
			if (i < ITERATIVE_COUNT - 1) {
				memcpy(input, output_gold, sizeof(T)*N*N*N);
			}
		}

		error = checkError3D<T, N>(N, N, (T*)output, (T*)output_gold, ITERATIVE_COUNT, N - ITERATIVE_COUNT, ITERATIVE_COUNT, N - ITERATIVE_COUNT, ITERATIVE_COUNT, N - ITERATIVE_COUNT);
		printf("N=%d, [Test] RMS Error : %e\n", N, error);

		delete[] input;
		delete[] output;
		delete[] output_gold;

		if (error > TOLERANCE)
			return -1;
	}
};


int stencil_3d7pt_v3(int argc, char** argv) {
	using namespace v3;
#if defined(_DEBUG) || defined(DEBUG)
	const int N = 32*2;
#else
	const int N = GRID_SIZE;
#endif
	if (IsTeslaV100()) {
		printf("use V100 GPU, ");
		const int B = 512;
		const int C = 12;
		//V100, It 3, 14 24
		//{
		//	const int ITERATIVE_COUNT = 2;
		//	_3d7pt<double, ITERATIVE_COUNT, N, B, C>(argc, argv);
		//	_3d7pt<float, ITERATIVE_COUNT, N, 512, 18>(argc, argv);
		//}
		{
			const int ITERATIVE_COUNT = 3;
			_3d7pt<double, ITERATIVE_COUNT, N, B, C>(argc, argv);
			_3d7pt<float, ITERATIVE_COUNT, N, 512, 16>(argc, argv);
		}
	} 
	else {
		printf("use P100 GPU, ");
		//For P100 and other GPUs
		const int B = 768;//512;
		const int C = 12;
		//P100  It 3, 512, 16
		//V100, It 3, 14 24
		{
			const int ITERATIVE_COUNT = 1;
			_3d7pt<double, ITERATIVE_COUNT, N, B, 16>(argc, argv);
			//_3d7pt<float, ITERATIVE_COUNT, N, 512, 18>(argc, argv);
		}
		{ 
		 	const int ITERATIVE_COUNT = 2; 
		 	_3d7pt<double, ITERATIVE_COUNT, N, 768, 20>(argc, argv); 
		 	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 18>(argc, argv);
		} 
		{
			const int ITERATIVE_COUNT = 2;
			_3d7pt<double, ITERATIVE_COUNT, N, 768, 14>(argc, argv);
			//_3d7pt<float, ITERATIVE_COUNT, N, 512, 18>(argc, argv);
		}
		{ 
		 	const int ITERATIVE_COUNT = 3; 
			_3d7pt<double, ITERATIVE_COUNT, N, 512, 16>(argc, argv); 
		 	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 18>(argc, argv); 
		} 
		{
		 	//const int ITERATIVE_COUNT = 4;
		 	//_3d7pt<double, ITERATIVE_COUNT, N, 1024, 12>(argc, argv);
		 	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 18>(argc, argv);
		}
		//{
		//	const int ITERATIVE_COUNT = 2;
		//	_3d7pt<double, ITERATIVE_COUNT, N, 512, 14>(argc, argv);
		//	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 18>(argc, argv);
		//}
		//{
		//	const int ITERATIVE_COUNT = 3;
		//	_3d7pt<double, ITERATIVE_COUNT, N, B, 14>(argc, argv);
		//	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 16>(argc, argv);
		//}
		//{
		//	const int ITERATIVE_COUNT = 2;
		//	_3d7pt<double, ITERATIVE_COUNT, N, 480, 15>(argc, argv);
		//	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 16>(argc, argv);
		//}
		//{
		//	const int ITERATIVE_COUNT = 3;
		//	_3d7pt<double, ITERATIVE_COUNT, N, 480, 15>(argc, argv);
		//	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 16>(argc, argv);
		//}
		//{
		//	const int ITERATIVE_COUNT = 4;
		//	_3d7pt<double, ITERATIVE_COUNT, N, 480, 15>(argc, argv);
		//	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 16>(argc, argv);
		//}
		//{
		//	const int ITERATIVE_COUNT = 4;
		//	_3d7pt<double, ITERATIVE_COUNT, N, B, 14>(argc, argv);
		//	//_3d7pt<float, ITERATIVE_COUNT, N, 512, 16>(argc, argv);
		//}

		//{
		//	const int ITERATIVE_COUNT = 2;
		//	_3d7pt<double, ITERATIVE_COUNT, N, B, 12>(argc, argv);
		//	_3d7pt<float, ITERATIVE_COUNT, N, 512, 19>(argc, argv);
		//}
		//{
		//	const int ITERATIVE_COUNT = 3;
		//	_3d7pt<double, ITERATIVE_COUNT, N, B, C>(argc, argv);
		//	_3d7pt<float, ITERATIVE_COUNT, N, 512, 16>(argc, argv);
		//}
		//dtype=double, all_time=5.901632ms, one_itr_time=2.950816ms, N=512,Iterative=2, CacheCount=12, WarpCount=16,  gups=45.48GUPS, gflop=591.30GFLOP/s
		//dtype=float, all_time=3.933504ms, one_itr_time=1.311168ms, N=512,Iterative=3, CacheCount=16, WarpCount=16,  gups=102.37GUPS, gflop=1330.75GFLOP/s
	} 

	return 0;
}
