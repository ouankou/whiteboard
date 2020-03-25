#ifndef __COMMON_H
#define __COMMON_H
#pragma once
#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <string>
#include <typeinfo>
#include <cstdlib>
#include <iostream>
#include <ctime>


template<typename T>
static T get_random() {
	return ((T)(rand()) / (T)(RAND_MAX - 1));
}

inline std::string FormatString(const char* fmt, ...) {
	char szBuf[1024 * 8] = "";
	va_list list;
	va_start(list, fmt);
	vsprintf(szBuf, fmt, list);
	va_end(list);
	return std::string(szBuf);
}

#define INSTANCE(func, type)\
	int func##_##type(int argc, char** argv){\
			return func<type>(argc, argv);\
		}

#define DISPLAY_FUNCTION(__comments) std::cout<<"call_function : "<<__FUNCTION__<<" : "<<__comments<<std::endl;

template<typename T>
struct ImageT {
	ImageT():data(NULL) {
	}
	inline ImageT& MallocBuffer(int w, int h) {
		_mem.resize(w*h);
		data = !_mem.empty() ? &_mem[0] : NULL;
		width = w;
		height = h;
		return *this;
	}
	ImageT& Clear() {
		if (data) {
			memset(data, 0, sizeof(data[0])*_mem.size());
		}
		return *this;
	}
	inline bool Save(const char* szPath) {
		bool bRtn = false;
		if (data) {
			FILE* fp = fopen(szPath, "wb");
			if (fp) {
				bRtn = width*height == fwrite(data, sizeof(data[0]), width*height, fp) ? true : false;
				fclose(fp);
			}
		}
		return bRtn;
	}
	template<typename D>
	inline bool Load(const char* szPath, int w, int h) {
		bool bRtn = false;
		std::vector<D> vec(w*h);
		FILE* fp = fopen(szPath, "rb");
		if (fp) {
			if (w*h == fread(&vec[0], sizeof(vec[0]), w*h, fp)) {
				width = w;
				height = h;
				bRtn = true;
			}
			fclose(fp);
		}
		_mem.clear();
		_mem.resize(vec.size());
		if (bRtn) {
			width = w;
			height = h;
			for (int i = 0; i < vec.size(); i++)
				_mem[i] = vec[i];
		}
		data = !_mem.empty() ? &_mem[0] : NULL;
		return bRtn;
	}
	T* data;
	int width, height;
private:
	std::vector<T> _mem;
};


template<typename T>
struct DataT {
	DataT(int w = 0, int h = 0, int c = 0, int n = 0) :data(NULL), width(w), height(h), channels(c), count(n) {
		if (w*h*c*n > 0) MallocBuffer(w, h, c, n);
	}
	inline DataT& MallocBuffer(int w, int h = 1, int c = 1, int n = 1) {
		_mem.resize(w*h*c*n);
		data = !_mem.empty() ? &_mem[0] : NULL;
		width = w;
		height = h;
		channels = c;
		count = n;
		return *this;
	}
	inline size_t Size() const {
		return width*height*channels*count;
	}
	inline bool SaveRaw(const char* szPath) {
		bool bRtn = false;
		if (data) {
			FILE* fp = fopen(szPath, "wb");
			if (fp) {
				bRtn = width*height*channels*count == fwrite(data, sizeof(data[0]), width*height*channels*count, fp) ? true : false;
				fclose(fp);
			}
		}
		printf("save:%s,%s\n", szPath, bRtn?"success":"failed");
		return bRtn;
	}
	void SaveText(const char* szPath) {
		printf("save : %s\n", szPath);
		FILE* fp = fopen(szPath, "wt");
		if (fp) {
			int s = 0;
			fprintf(fp, "{\n");
			for (int i = 0; i < count; i++) {
				fprintf(fp, " {\n");
				for (int j = 0; j < channels; j++) {
					fprintf(fp, "  {\n");
					for (int m = 0; m < height; m++) {
						fprintf(fp, "   {");
						for (int n = 0; n < width; n++) {
							fprintf(fp, "%f ", float(data[s++]));
						}
						fprintf(fp, "   }\n");
					}
					fprintf(fp, "   }\n");
				}
				fprintf(fp, " }\n");
			}
			fprintf(fp, "}\n");
			fclose(fp);
		}
	}
	inline bool Load_uchar(const char* szPath, int w, int h = 1, int c = 1, int n = 1){
		return Load<unsigned char>(szPath, w, h, c, n);
	}
	template<typename D>
	inline bool Load(const char* szPath, int w, int h=1, int c=1, int n=1) {
		bool bRtn = false;
		std::vector<D> vec(w*h*c*n);
		FILE* fp = fopen(szPath, "rb");
		if (fp) {
			if (w*h == fread(&vec[0], sizeof(vec[0]), w*h*c*n, fp)) {
				width = w;
				height = h;
				channels = c;
				count = n;
				bRtn = true;
			}
			fclose(fp);
		}
		_mem.clear();
		_mem.resize(vec.size());
		if (bRtn) {
			width = w;
			height = h;
			channels = c;
			count = n;
			for (int i = 0; i < vec.size(); i++)
				_mem[i] = vec[i];
		}
		data = !_mem.empty() ? &_mem[0] : NULL;
		return bRtn;
	}
	T* data;
	int width, height, channels, count;
private:
	std::vector<T> _mem;
};


template<typename T> inline
void Convolution(const T* src, T* dst, int width, int height, const T* filter, int flt_width, int flt_height)
{
	int flt_size = flt_width*flt_height;

#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		//T* buf = new T[flt_size];
		for (int j = 0; j < width; j++) {
			//memset(buf, 0, sizeof(buf[0])*flt_size);

			T sum = 0;
			for (int n = 0; n < flt_width; n++) {
				for (int m = 0; m < flt_height; m++) {
					int x = j + n - flt_width / 2;
					int y = i + m - flt_height / 2;
					if (x >= 0 && x < width && y >= 0 && y < height) {
						//buf[m*flt_width + n] = src[y*width + x];
						int idx = m*flt_width + n;
						sum += src[y*width + x] * filter[idx];
					}
				}
			}
			//for (int m = 0; m < flt_size; m++)
			//	sum += buf[m] * filter[m];
			dst[i*width + j] = sum;
		}
		//if (buf) delete []buf;
	}

}

template<typename T> inline
double Convolution_verify(const T* src, const T* dst, T* workBuf, int width, int height, const T* filter, int flt_width, int flt_height)
{
	T* pImg = workBuf;

	Convolution(src, pImg, width, height, filter, flt_width, flt_height);
	double dif = 0;
	for (int i = flt_height; i < height - flt_height; i++) {
		for (int j = flt_width; j < width - flt_width; j++) {
			int offset = j + i * width;
			dif += fabs(double(dst[offset] - pImg[offset]));
		}
	}
	return dif;
}

#define Batch_Def_1( __Type_Pre_One, x0)     __Type_Pre_One(x0)
#define Batch_Def_2( __Type_Pre_One, x0, x1) Batch_Def_1( __Type_Pre_One, x0) Batch_Def_1( __Type_Pre_One, x1)
#define Batch_Def_3( __Type_Pre_One, x0, x1, x2) Batch_Def_2(__Type_Pre_One, x0, x1) Batch_Def_1(__Type_Pre_One, x2)
#define Batch_Def_4( __Type_Pre_One, x0, x1, x2, x3) Batch_Def_3( __Type_Pre_One, x0, x1, x2) Batch_Def_1(__Type_Pre_One, x3)
#define Batch_Def_5( __Type_Pre_One, x0, x1, x2, x3, x4) Batch_Def_4( __Type_Pre_One, x0, x1, x2, x3) Batch_Def_1(__Type_Pre_One, x4)
#define Batch_Def_6( __Type_Pre_One, x0, x1, x2, x3, x4, x5) Batch_Def_5( __Type_Pre_One, x0, x1, x2, x3, x4) Batch_Def_1(__Type_Pre_One, x5)
#define Batch_Def_7( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6) Batch_Def_6( __Type_Pre_One, x0, x1, x2, x3, x4, x5) Batch_Def_1(__Type_Pre_One, x6)
#define Batch_Def_8( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7) Batch_Def_7( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6) Batch_Def_1(__Type_Pre_One, x7)
#define Batch_Def_9( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8) Batch_Def_8( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7) Batch_Def_1( __Type_Pre_One, x8)
#define Batch_Def_10(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9) Batch_Def_9( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8) Batch_Def_1( __Type_Pre_One, x9)
#define Batch_Def_11(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) Batch_Def_10( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9) Batch_Def_1( __Type_Pre_One, x10)
#define Batch_Def_12(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) Batch_Def_11( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) Batch_Def_1( __Type_Pre_One, x11)
#define Batch_Def_13(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) Batch_Def_12( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) Batch_Def_1( __Type_Pre_One, x12)
#define Batch_Def_14(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) Batch_Def_13( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) Batch_Def_1( __Type_Pre_One, x13)
#define Batch_Def_15(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14) Batch_Def_14( __Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) Batch_Def_1( __Type_Pre_One, x14)
#define Batch_Def_16(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) Batch_Def_8(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7) Batch_Def_8(__Type_Pre_One, x8, x9, x10, x11, x12, x13, x14, x15)
#define Batch_Def_32(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31) Batch_Def_16(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) Batch_Def_16(__Type_Pre_One, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31)
#define Batch_Def_64(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63) Batch_Def_32(__Type_Pre_One, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31) Batch_Def_32(__Type_Pre_One, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63)



#endif //__COMMON_H
