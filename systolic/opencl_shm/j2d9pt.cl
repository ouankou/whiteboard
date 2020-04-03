
#define DataType float4


__kernel void j2d9pt(__global float4 * dst, __global float4 * src, __local float4 * block)
{
    const int width = 256;
    const int height = 256;
    const int BLOCK_SIZE = 128;
    const int PROCESS_DATA_COUNT = 4;
    const int WARP_SIZE = 32;
    const int FILTER_WIDTH = 5;
    const int FILTER_HEIGHT = 5;
    const int WARP_COUNT = BLOCK_SIZE >> 5;
    const int laneId = get_local_id(0) & 31;
    const int warpId = get_local_id(0) >> 5;
    const int WARP_PROCESS_DATA_COUNT = WARP_SIZE - FILTER_WIDTH + 1;
    const int BLOCK_PROCESS_DATA_COUNT = WARP_PROCESS_DATA_COUNT*WARP_COUNT;
    const int DATA_CACHE_SIZE = PROCESS_DATA_COUNT + FILTER_HEIGHT - 1;

    DataType fc = 5;
    DataType fn0 = 2;
    DataType fs0 = 8;
    DataType fw0 = 4;
    DataType fe0 = 6;
    DataType fn1 = 1;
    DataType fs1 = 9;
    DataType fw1 = 3;
    DataType fe1 = 7;


    //printf("KERNEL SRC:\n");
    //for(int i = 0; i < height; i++)
    //{
    //    for(int j = 0; j < width; j++)
    //    {
    //        printf("%f, ", src[i*width+j]);
    //    }
    //    printf("\n");
    //}
    //printf("\n");

    //DataType data[DATA_CACHE_SIZE];
    DataType data[8];
    //__local DataType shared_sum[BLOCK_SIZE];
    __local DataType shared_sum[128];
    //const int sumId = warpId*WARP_SIZE + laneId;
    const int sumId = get_local_id(0);

    const int process_count = BLOCK_PROCESS_DATA_COUNT*get_group_id(0) + WARP_PROCESS_DATA_COUNT*warpId;
    if (process_count >= width)
        return;
    int tidx = process_count + laneId - FILTER_WIDTH / 2;
    int tidy = PROCESS_DATA_COUNT*get_group_id(1) - FILTER_HEIGHT / 2;

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
        //T sum = data[i + 2] * fe1;
        shared_sum[sumId] = data[i + 2] * fe1;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        //sum = __my_shfl_down(sum, 1);
        if (laneId != 31) 
        shared_sum[sumId] = shared_sum[(sumId+1)];
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        //sum += data[i + 2] * fe0;
        shared_sum[sumId] += data[i + 2] * fe0;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();

        //sum = __my_shfl_down(sum, 1);
        if (laneId != 31) 
        shared_sum[sumId] = shared_sum[(sumId+1)];
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        shared_sum[sumId] += data[i + 0] * fn1;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        shared_sum[sumId] += data[i + 1] * fn0;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        shared_sum[sumId] += data[i + 2] * fc;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        shared_sum[sumId] += data[i + 3] * fs0;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        shared_sum[sumId] += data[i + 4] * fs1;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();

        //sum = __my_shfl_down(sum, 1);
        if (laneId != 31) 
        shared_sum[sumId] = shared_sum[(sumId+1)];
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        shared_sum[sumId] += data[i + 2] * fw0;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();

        //sum = __my_shfl_down(sum, 1);
        if (laneId != 31) 
        shared_sum[sumId] = shared_sum[(sumId+1)];
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
        shared_sum[sumId] += data[i + 2] * fw1;
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();

        data[i] = shared_sum[sumId];
        barrier(CLK_LOCAL_MEM_FENCE);
        //__syncwarp();
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

