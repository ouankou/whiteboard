#include "axpy.h"

#ifdef __cplusplus
extern "C" {
#endif

//This two global variables are required by libomptarget runtime
//The first one is automatically added by the clang/llvm compiler
__device__ char axpy_cudakernel_1perThread_exec_mode = 0; //<kernel_name>_exec_mode, added by the compiler
__device__ int32_t omptarget_device_environment; //in libomptarget-nvptx.a library

__global__ void
axpy_cudakernel_1perThread(REAL* y, REAL *x, int n, REAL a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] += a*x[i];
}
#ifdef __cplusplus
}
#endif
