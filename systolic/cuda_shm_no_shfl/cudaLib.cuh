#ifndef __cudaLib_H
#define __cudaLib_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <assert.h>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

#define MyFORCE_CHECK_CUDA(call)  do {                                    \
    const cudaError_t status = call;                                    \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

#ifndef FORCE_CHECK_CUDA
#define FORCE_CHECK_CUDA
#endif

#ifdef DEBUG
#define CHECK_CUDA(call)  FORCE_CHECK_CUDA(call)
#else
#define CHECK_CUDA(call)  (call)
#endif

#define __cudaCheckError(x)  if (cudaSuccess != (x)){          \
	printf("file %s, line %d, function %s,\nCUDA ERROR:%s\n",  \
	__FILE__, __LINE__, __FUNCTION__, cudaGetErrorString(x));  \
	assert(0);throw(std::string(cudaGetErrorString(x)));       \
}

#define __cudaSafeCall(x)    if (cudaSuccess != (x)){__cudaCheckError((x));}


#define CUDA_CHECK_ERROR { cudaError_t err = cudaGetLastError(); __cudaCheckError(err); }

//convert const pointer to general one
template<typename T> inline
T* MyConstToGeneral(const T* ptr){ T* _ptr; memcpy(&_ptr, &ptr, sizeof(_ptr)); return _ptr; }

#ifndef ConstToGeneral
#define ConstToGeneral MyConstToGeneral
#endif

#ifndef UpDivide
#define UpDivide MyUpDivide
#endif

#ifndef UpRound
#define UpRound MyUpRound
#endif

__device__ __host__ __forceinline__ unsigned int MyUpDivide(unsigned int x, unsigned int y){ assert(y!=0); return (x+y-1)/y; }
__device__ __host__ __forceinline__ unsigned int MyUpRound(unsigned int x, unsigned int y) { return UpDivide(x, y)*y;}


template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

#ifndef DevStream
#define DevStream MyDevStream
#endif

struct MyDevStream{
	MyDevStream():stream(cudaStreamDefault){
	}
	~MyDevStream(){
		Destroy();
	}
	inline MyDevStream& Create(){
		Destroy();
		cudaStreamCreate(&stream);
		CUDA_CHECK_ERROR;
		return *this;
	}
	inline void Destroy(){
		if (!IsDefault()) cudaStreamDestroy(stream);
	}
	inline bool IsDefault() const{
		return stream == cudaStreamDefault?true:false;
	}
	cudaStream_t stream;
};


template<class T, int DIM>
struct DevDataPtr{
	T* data;
	size_t width, data_pitch, height, depth;
};


#define TYPE_DevDataPtr1(type)\
	template<> struct DevDataPtr<type, 1>{\
	__device__ __host__ __forceinline__ DevDataPtr(type* _data, size_t _width):data(_data), width(_width){}\
	__device__ __host__ __forceinline__ float& operator()(size_t x) const{\
		return data[x];\
	}\
	__device__ __host__ __forceinline__ DevDataPtr& operator=(const DevDataPtr& obj){\
		data = obj.data;width = obj.width;\
		return *this;\
	}\
	type* data;\
	size_t width;\
};

#define TYPE_DevDataPtr2(type)\
	template<>struct DevDataPtr<type, 2>{\
	__device__ __host__ __forceinline__ DevDataPtr(type* _data, size_t _width, size_t _data_pitch, size_t _height)\
		:data(_data), width(_width), data_pitch(_data_pitch), height(_height){\
	}\
	__device__ __host__ __forceinline__ float& operator()(size_t x, size_t y) const{\
		return data[x + y*data_pitch];\
	}\
	template<typename _Tp>\
	__device__ __host__ __forceinline__ bool IsValid(_Tp x, _Tp y) const{\
		return x>=0 && x<=width-1 && y>=0 && y<=height-1;\
	}\
	__device__ __host__ __forceinline__ DevDataPtr& operator=(const DevDataPtr& obj){\
		data = obj.data;width = obj.width; height = obj.height;\
		return *this;\
	}\
	type* data;\
	size_t width, data_pitch, height;\
};

#define TYPE_DevDataPtr3(type)\
	template<>struct DevDataPtr<type, 3>{\
	__device__ __host__ __forceinline__ DevDataPtr(type* _data, size_t _width, size_t _data_pitch, size_t _height, size_t _depth)\
		:data(_data), width(_width), data_pitch(_data_pitch), height(_height), depth(_depth){\
	}\
	__device__ __host__ __forceinline__ float& operator()(size_t x, size_t y, size_t z) const{\
		return data[x + (y + z*height)*data_pitch];\
	}\
	template<typename _Tp>\
	__device__ __host__ __forceinline__ bool IsValid(_Tp x, _Tp y, _Tp z) const{\
		return x>=0 && x<=width-1 && y>=0 && y<=height-1 && z>=0 && z<=depth-1;\
	}\
	__device__ __host__ __forceinline__ DevDataPtr& operator=(const DevDataPtr& obj){\
		data = obj.data;width = obj.width; height = obj.height; depth = obj.depth;\
		return *this;\
	}\
	type* data;\
	size_t width, data_pitch, height, depth;\
};

TYPE_DevDataPtr1(float);
TYPE_DevDataPtr2(float);
TYPE_DevDataPtr3(float);





template<class T>
struct MyDevDataRef{
	typedef DevDataPtr<T, 1> Ptr1D;
	typedef DevDataPtr<T, 2> Ptr2D;
	typedef DevDataPtr<T, 3> Ptr3D;

	inline Ptr1D GetPtr1D() const{
		return Ptr1D(data, width);
	}
	inline Ptr2D GetPtr2D() const{
		return Ptr2D(data, width, data_pitch, height*depth);
	}
	inline Ptr3D GetPtr3D() const{
		return Ptr3D(data, width, data_pitch, height, depth);
	}
	inline MyDevDataRef(){
		width = height = depth = data_pitch;
		data = 0;
	}
	inline MyDevDataRef(T* ptr, size_t w, size_t pitch, size_t h, size_t d){
		data = ptr; width = w; height = h; depth = d; data_pitch = pitch;
	}
	inline MyDevDataRef& operator=(const MyDevDataRef& obj){
		data = obj.data; width = obj.width; height = obj.height; depth = obj.depth; data_pitch = obj.data_pitch;
		return *this;
	}
	inline void Zero(cudaStream_t stream = cudaStreamDefault){
		if (stream == cudaStreamDefault)
			cudaMemset2D(data, data_pitch*sizeof(*data), 0, width*sizeof(*data), height*depth);
		else
			cudaMemset2DAsync(data, data_pitch*sizeof(*data), 0, width*sizeof(*data), height*depth, stream);
	}
	inline void Ones(cudaStream_t stream = cudaStreamDefault){
		SetValue(T(1), stream);
	}
	inline void SetValue(T val, cudaStream_t stream = cudaStreamDefault){
		size_t size = width*height*depth;
		std::vector<T> vec(size + 1);
		T* buf = & vec[0];
		for (size_t i=0; i<size; i ++) buf[i] = val;
		this->CopyFromHost(buf, width, width, height, depth, stream);
	}
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault) const{
		assert(_w <= width && _h*_d <= height*depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(ptr, sizeof(*ptr)*_pitch, data, data_pitch*sizeof(*data), width*sizeof(*data), _h*_d, cudaMemcpyDeviceToHost);	
		else
			cudaMemcpy2DAsync(ptr, sizeof(*ptr)*_pitch, data, data_pitch*sizeof(*data), width*sizeof(*data), _h*_d, cudaMemcpyDeviceToHost, stream);	
		CUDA_CHECK_ERROR;
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault) const{
		assert(_w <= width && _h*_d <= height*depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(ptr, sizeof(*ptr)*_pitch, data, data_pitch*sizeof(*data), width*sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpy2DAsync(ptr, sizeof(*ptr)*_pitch, data, data_pitch*sizeof(*data), width*sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice, stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault){
		assert(_w <= width && _h*_d <= height*depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(data, data_pitch*sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w*sizeof(*data), _h*_d, cudaMemcpyHostToDevice);
		else
			cudaMemcpy2DAsync(data, data_pitch*sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w*sizeof(*data), _h*_d, cudaMemcpyHostToDevice, stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d, cudaStream_t stream = cudaStreamDefault){
		assert(_w <= width && _h*_d <= height*depth && _w <= _pitch);
		if (stream == cudaStreamDefault)
			cudaMemcpy2D(data, data_pitch*sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w*sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpy2DAsync(data, data_pitch*sizeof(*data), ptr, sizeof(*ptr)*_pitch, _w*sizeof(*data), _h*_d, cudaMemcpyDeviceToDevice, stream);
		CUDA_CHECK_ERROR;
	}
	inline bool BindToTexture1D(const textureReference* _tex, size_t offset = 0, cudaChannelFormatDesc* pDesc = NULL) const{
		bool bRtn = false;
		if (data){
			cudaChannelFormatDesc desc = pDesc?*pDesc:cudaCreateChannelDesc<T> ();
			cudaBindTexture(&offset, _tex, data, &desc, width);		
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	inline bool BindToTexture2D(const textureReference* _tex, size_t offset = 0, cudaChannelFormatDesc* pDesc = NULL) const{
		bool bRtn = false;
		if (data){
			cudaChannelFormatDesc desc = pDesc?*pDesc:cudaCreateChannelDesc<T> ();
			cudaBindTexture2D(&offset, _tex, data, &desc, width, height, data_pitch*sizeof(*data));			
			CUDA_CHECK_ERROR;
			bRtn = true;
		}
		return bRtn;
	}
	inline bool UnbindTexture(const textureReference* _tex){
		cudaError err = cudaUnbindTexture(_tex);
		CUDA_CHECK_ERROR;
		return err == cudaSuccess?true:false;
	}
	void Display(std::string name, size_t z = 0) const{
		if (z<0 || z >=depth){
			assert(0);
			return;
		}
		const size_t size = width*height*depth;
		const size_t offset = z*width*height;
		std::vector<T> tmp(size+1);
		T* buf = &tmp[0];
		this->CopyToHost(buf, width, width, height, depth);
		buf += offset;
	}
	T* data;
	size_t width, data_pitch, height, depth;
	std::vector<std::string> vecTexSymbol;
};

#ifndef DevData
#define DevData MyDevData
#endif

template<class T>
struct MyDevData {
	typedef T DataType;
	typedef MyDevDataRef<T> DataRef;
	inline virtual ~MyDevData(){
		this->FreeBuffer();
	}
	inline MyDevData():width(0),height(0), depth(0),channels(0){
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
	}
	inline MyDevData(size_t w, size_t h=1, size_t d=1):width(0),height(0), depth(0),channels(0){
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
		MallocBuffer(w, h, d);
	}
	inline MyDevData& operator=(const MyDevData& obj){
		MallocBuffer(obj.width, obj.height, obj.depth);
		GetDataRef().CopyFromDevice(obj.GetData(), obj.width, obj.DataPitch(), obj.height, obj.depth, stream.stream);
		return *this;
	}
	inline DataRef GetDataRef() const{
		return DataRef(GetData(), width, DataPitch(), height, depth); 
	}
	inline MyDevData& MallocBuffer(size_t w, size_t h, size_t d){
		assert(h*d >= 1);
		if (width == w && height == h && depth == d){
		}else{
			//reuse the device aligned buffer
			if (sizeof(T)*w < this->dev_data.pitch && h*d > 0 && h*d <= this->extent.height*this->extent.depth){
				width = w;
				height = h;
				depth = d;
			}else{
				FreeBuffer();
				extent = make_cudaExtent(sizeof(T)*w, h, d);
				cudaMalloc3D(&dev_data, extent);
				CUDA_CHECK_ERROR;
				width = w;
				height = h;
				depth = d;
				if (d == 1){
					if (h == 1) channels = 1;
					else        channels = 2;	
				}else           channels = 3;	
			}
		}
		return *this;
	}
	inline MyDevData& MallocBuffer(size_t w, size_t h){
		return MallocBuffer(w, h, 1);
	}
	inline MyDevData& MallocBuffer(size_t w){
		return MallocBuffer(w, 1, 1);
	}
	inline void FreeBuffer(){
		if (dev_data.ptr){
			cudaFree(dev_data.ptr);
		}
		memset(&dev_data, 0, sizeof(dev_data));
		memset(&extent, 0, sizeof(extent));
		width = height = depth = 0;
		channels = 0;
	}
	inline bool BindToTexture1D(const textureReference* _tex, cudaChannelFormatDesc* pDesc = NULL) const{
		size_t offset = 0;
		return this->GetDataRef().BindToTexture1D(_tex, offset, pDesc);
	}
	inline bool BindToTexture2D(const textureReference* _tex, int nDepthIndex = 0, cudaChannelFormatDesc* pDesc = NULL) const{
		bool bRtn = false;
		size_t offset = this->DataPitch()*height*nDepthIndex;
		bRtn = this->GetDataRef().BindToTexture2D(_tex, offset, pDesc);
		return bRtn;
	}
	inline bool UnbindTexture(const textureReference* _tex) const{
		return this->GetDataRef().UnbindTexture(_tex);
	}
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d=1){
		GetDataRef().CopyToHost(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d=1){
		GetDataRef().CopyToDevice(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d=1){
		 GetDataRef().CopyFromHost(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d=1){
		GetDataRef().CopyFromDevice(ptr, _w, _pitch, _h, _d, stream.stream);
	}
	inline MyDevData& Zero(){
		GetDataRef().Zero(stream.stream);
		return *this;
	}
	inline MyDevData& Ones(){
		GetDataRef().Ones(stream.stream);
		return *this;
	}
	inline MyDevData& SetValue(T val){
		GetDataRef().SetValue(val, stream.stream);
		return *this;
	}
	inline size_t DataPitch() const{
		return dev_data.pitch/sizeof(T);
	}
	inline T* GetData() const{
		return (T*)dev_data.ptr;
	}
	inline void Display(std::string name) const{
		GetDataRef().Display(name);
	}
	inline MyDevData& SetStream(const MyDevStream& s){
		stream.stream = s.stream;
		return *this;
	}
	MyDevStream stream;
	cudaPitchedPtr dev_data;
	cudaExtent extent;
	size_t width, height, depth, channels;
};




template<class T>
struct DevBuffer {
	typedef T DataType;
	typedef MyDevDataRef<T> DataRef;
	inline virtual ~DevBuffer() {
		this->FreeBuffer();
	}
	inline DevBuffer() :width(0), height(0), depth(0), channels(0), data(NULL){
	}
	inline DevBuffer(size_t w, size_t h = 1, size_t d = 1) : width(0), height(0), depth(0), channels(0), data(NULL) {
		MallocBuffer(w, h, d);
	}
	inline DevBuffer& operator=(const DevBuffer& obj) {
		MallocBuffer(obj.width, obj.height, obj.depth);
		GetDataRef().CopyFromDevice(obj.GetData(), obj.width, obj.DataPitch(), obj.height, obj.depth, stream.stream);
		return *this;
	}
	inline DataRef GetDataRef() const {
		return DataRef(GetData(), width, DataPitch(), height, depth);
	}
	inline DevBuffer& MallocBuffer(size_t w, size_t h, size_t d) {
		assert(h*d >= 1);
		if (width == w && height == h && depth == d) {
		}
		else {
			size_t _sz = width*height*depth;
			size_t sz = w*h*d;
			if (_sz >= sz) {
			}else{
				FreeBuffer();
				cudaMalloc(&data, sz*sizeof(T));
				CUDA_CHECK_ERROR;
			}
			width = w;
			height = h;
			depth = d;
			channels = 1;
		}
		return *this;
	}
	inline DevBuffer& MallocBuffer(size_t w, size_t h) {
		return MallocBuffer(w, h, 1);
	}
	inline DevBuffer& MallocBuffer(size_t w) {
		return MallocBuffer(w, 1, 1);
	}
	inline void FreeBuffer() {
		if (data) cudaFree(data);
		data = NULL;
		width = height = depth = 0;
		channels = 0;
	}
	inline bool BindToTexture1D(const textureReference* _tex, cudaChannelFormatDesc* pDesc = NULL) const;
	inline bool BindToTexture2D(const textureReference* _tex, int nDepthIndex = 0, cudaChannelFormatDesc* pDesc = NULL);
	inline bool UnbindTexture(const textureReference* _tex) const;
	inline void CopyToHost(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToHost);
		else
			cudaMemcpyAsync(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToHost, stream.stream);
	}
	inline void CopyToDevice(T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpyAsync(ptr, data, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice, stream.stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromHost(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyHostToDevice);
		else
			cudaMemcpyAsync(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyHostToDevice, stream.stream);
		CUDA_CHECK_ERROR;
	}
	inline void CopyFromDevice(const T* ptr, size_t _w, size_t _pitch, size_t _h, size_t _d = 1) {
		assert(_w == _pitch);
		if (stream.stream == cudaStreamDefault)
			cudaMemcpy(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpyAsync(data, ptr, sizeof(T)*_w*_h*_d, cudaMemcpyDeviceToDevice, stream.stream);
		CUDA_CHECK_ERROR;
	}
	inline DevBuffer& Zero() {
		GetDataRef().Zero(stream.stream);
		return *this;
	}
	inline DevBuffer& Ones() {
		GetDataRef().Ones(stream.stream);
		return *this;
	}
	inline DevBuffer& SetValue(T val) {
		GetDataRef().SetValue(val, stream.stream);
		return *this;
	}
	inline size_t DataPitch() const {
		return width;
	}
	inline T* GetData() const {
		return (T*)data;
	}
	inline void Display(std::string name) const {
		GetDataRef().Display(name);
	}
	inline DevBuffer& SetStream(const MyDevStream& s) {
		stream.stream = s.stream;
		return *this;
	}
	MyDevStream stream;
	T* data;
	size_t width, height, depth, channels;
};

template<typename T> inline
cudaError_t MyLoadConstant(const void* cData, const T* data, int size){
	cudaError_t err = cudaMemcpyToSymbol(cData, data, sizeof(data[0])*size);
	CUDA_CHECK_ERROR;
	return err;
}

template<typename T> inline
cudaError_t MyLoadConstant(const MyDevStream& sm, const void* cData, const T* data, int size){
	cudaError_t err = cudaSuccess;
	int offset = 0;
	if (sm.stream == cudaStreamDefault) err = cudaMemcpyToSymbol(cData, data, sizeof(data[0])*size, offset, cudaMemcpyHostToDevice);
	else err = cudaMemcpyToSymbolAsync(cData, data, sizeof(data[0])*size, offset, cudaMemcpyHostToDevice, sm.stream);
	CUDA_CHECK_ERROR;
	return err;
}

#ifndef NOUSE_MAD
template<typename T> __device__ __forceinline__ T MAD(T a, T b, T c);

template<> __device__ __forceinline__ int MAD<int>(int a, int b, int c) {
	asm("mad.wide.s32 %0, %1, %2, %3;" : "=r"(c) : "r"(a), "r"(b), "r"(c));
	return c;
}

template<> __device__ __forceinline__  uint MAD<uint>(uint a, uint b, uint c) {
	asm("mad.wide.u32 %0, %1, %2, %3;" : "=r"(c) : "r"(a), "r"(b), "r"(c));
	return c;
}

template<> __device__ __forceinline__  float MAD<float>(float a, float b, float c) {
	float d;
	asm volatile("mad.rz.ftz.f32 %0,%1,%2,%3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
	return d;
}
#endif //NOUSE_MAD

#if (CUDART_VERSION >= 9000)
#pragma message("CUDART_VERSION >= 9000")
#define __my_shfl_up(var, delta) __shfl_up_sync(0xFFFFFFFF, var, delta)
#define __my_shfl_down(var, delta) __shfl_down_sync(0xFFFFFFFF, var, delta)
#else
#pragma message("CUDART_VERSION < 9000")
#define __my_shfl_up(var, delta) __shfl_up(var, delta)
#define __my_shfl_down(var, delta) __shfl_down(var, delta)
#endif
#endif //__cudaLib_H
