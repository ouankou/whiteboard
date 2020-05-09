/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    mySum += g_idata[i];

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    sdata[tid] = mySum = mySum + sdata[tid + 64];
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}


extern "C" bool isPow2(unsigned int x);

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void reduce(int size, int threads, int blocks, int whichKernel, T *d_idata,
            T *d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
  //reduction using shuffle
      if (isPow2(size)) {
        switch (threads) {
          case 512:
            reduce6<T, 512, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 256:
            reduce6<T, 256, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 128:
            reduce6<T, 128, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 64:
            reduce6<T, 64, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 32:
            reduce6<T, 32, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 16:
            reduce6<T, 16, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 8:
            reduce6<T, 8, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 4:
            reduce6<T, 4, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 2:
            reduce6<T, 2, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 1:
            reduce6<T, 1, true>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
      } else {
        switch (threads) {
          case 512:
            reduce6<T, 512, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 256:
            reduce6<T, 256, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 128:
            reduce6<T, 128, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 64:
            reduce6<T, 64, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 32:
            reduce6<T, 32, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 16:
            reduce6<T, 16, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 8:
            reduce6<T, 8, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 4:
            reduce6<T, 4, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 2:
            reduce6<T, 2, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;

          case 1:
            reduce6<T, 1, false>
                <<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
            break;
        }
      }
}

// Instantiate the reduction function for 3 types
template void reduce<int>(int size, int threads, int blocks, int whichKernel,
                          int *d_idata, int *d_odata);

template void reduce<float>(int size, int threads, int blocks, int whichKernel,
                            float *d_idata, float *d_odata);

template void reduce<double>(int size, int threads, int blocks, int whichKernel,
                             double *d_idata, double *d_odata);

#endif  // #ifndef _REDUCE_KERNEL_H_
