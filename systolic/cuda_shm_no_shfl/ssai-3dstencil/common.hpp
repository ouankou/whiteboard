#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef GRID_SIZE
#define GRID_SIZE  (512)
//#pragma message("## GRID_SIZE = 512")
#else
#if GRID_SIZE <= 0
#error (error grid size definition)
#endif
#endif



#ifdef _WIN32
#else
#include <sys/time.h>
#endif

#if DEBUG == 1
	#ifndef _DEBUG
		#define _DEBUG
	#endif
#endif

#define TOLERANCE 1e-5

static const char* GetDeviceName(int dev = 0) {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	assert(dev < nDevices);
	static cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, dev);
	return prop.name;
}

static bool IsTeslaP100(int dev = 0)
{
	if (strstr(GetDeviceName(dev), "P100"))
		return true;
	return false;
}

static bool IsTeslaV100(int dev = 0)
{
	if (strstr(GetDeviceName(dev), "V100"))
		return true;
	return false;
}

template<typename T>
	static T get_random() {
		return ((T)(rand()) / (T)(RAND_MAX - 1));
	}

template<typename T>
	static T* getRandom3DArray(int height, int width_y, int width_x) {
		T *a = new T[height*width_y*width_x];
		T* p = a;
		int c = 0;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width_y; j++)
				for (int k = 0; k < width_x; k++) {
					p[c] = get_random<T>() + 0.02121;
					//p[c] = c % (width_x*width_y); //c++%1024;
					c++;
				}
		return (T*)a;
	}


template<typename T>
	static T* getZero3DArray(int height, int width_y, int width_x) {
		T(*a) = new T[height * width_y*width_x];
		memset((void*)a, 0, sizeof(T) * height * width_y * width_x);
		return (T*)a;
	}

template<typename T, int _N>
	static double checkError3D
	(int width_y,
		int width_x,
		const T *l_output,
		const T *l_reference,
		int z_lb,
		int z_ub,
		int y_lb,
		int y_ub,
		int x_lb,
		int x_ub) {
		const T(*output)[_N][_N] = (const T(*)[_N][_N])(l_output);
		const T(*reference)[_N][_N] = (const T(*)[_N][_N])(l_reference);
		double error = 0.0;
		double max_error = TOLERANCE, sum = 0.0;
		int max_k = 0, max_j = 0, max_i = 0, unmatch = 0, match = 0;
		int count = 0;
		for (int i = z_lb; i < z_ub; i++)
			for (int j = y_lb; j < y_ub; j++)
				for (int k = x_lb; k < x_ub; k++) {
					sum += output[i][j][k];
					//printf ("real var1[%d][%d][%d] = %.6f and %.6f\n", i, j, k, reference[i][j][k], output[i][j][k]);
					double curr_error = output[i][j][k] - reference[i][j][k];
					curr_error = (curr_error < 0.0 ? -curr_error : curr_error);
					if (curr_error == 0)
						match += 1;
					else 
						unmatch += 1;
					error += curr_error * curr_error;
					if (curr_error > max_error) {
						printf("(%d) Values at index (%d,%d,%d) differ : %.6f and %.6f\n", count++, i, j, k, reference[i][j][k], output[i][j][k]);
						max_error = curr_error;
						max_k = k;
						max_j = j;
						max_i = i;
					}
				}
		printf("sum = %e, match = %d, unmatch = %d\n", sum, match, unmatch);
		printf
			("[Test] Max Error : %e @ (%d,%d,%d)\n", max_error, max_i, max_j, max_k);
		error = sqrt(error / ((z_ub - z_lb) * (y_ub - y_lb) * (x_ub - x_lb)));
		return error;
	}

#endif
