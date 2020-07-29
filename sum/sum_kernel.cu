#include "sum.h"
#include <stdio.h>
#define BLOCK_NUM 1024
#define BLOCK_SIZE 256

__global__ 
void
global_1perThread(REAL* data, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = n/2;
    if (n%2 == 1) {
        stride += 1;
    }
    if (i + stride < n) {
       //printf("Num id: %d, %d\n", i, i+stride);
       data[i] += data[i + stride];
    };
}

/* block distribution of loop iteration */
__global__ 
void global_block(REAL* data, REAL* output, int n) {
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = gridDim.x * blockDim.x;

	int element_per_thread = n / total_threads;
    int residue = n % total_threads, start_index, end_index;
    if (thread_id < residue) {
        element_per_thread += 1;
        start_index = element_per_thread * thread_id;
    }
    else {
        start_index = (element_per_thread + 1) * residue + element_per_thread * (thread_id - residue);
    };
	
	end_index = start_index + element_per_thread;
    if (end_index > n || (end_index == n && element_per_thread == 0)) {
        end_index = -1;
    };
	int i;
    if (end_index != -1) output[thread_id] = 0.0;
    for (i = start_index; i < end_index; i++) {
        output[thread_id] += data[i];
	}
    //if (end_index != -1) printf("Local res %d - %d, %d: %g\n", thread_id, start_index, end_index, output[thread_id]);
}

/* cyclic distribution of loop distribution */
__global__
void global_cyclic(REAL* data, REAL* output, int n) {
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = gridDim.x * blockDim.x;
	
	int element_per_thread = n / total_threads;
    int residue = n % total_threads;
    if (thread_id < residue) {
        element_per_thread += 1;
    };
	
	int i;
    if (thread_id < n) output[thread_id] = 0.0;
    for (i = thread_id; i < n; i += total_threads) {
        if (i < n) {
            output[thread_id] += data[i];
        }
        else {
            break;
        };
	}
}

void final_reduce(REAL* data_device, int n) {
    int residue;
    if (n/BLOCK_NUM*BLOCK_SIZE == 0) {
        residue = n;
    }
    else {
        residue = n%(BLOCK_NUM*BLOCK_SIZE);
    };
    while (residue > 1) {
        global_1perThread<<<(residue+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(data_device, residue);
        if (residue%2 == 1) {
            residue = residue/2 + 1;
        }
        else {
            residue /= 2;
        };
    };
}


REAL sum_kernel(REAL* input, int n, int kernel) {
    REAL *data_device, *output_device, result = 0.0;
    cudaMalloc(&data_device, n*sizeof(REAL));
    cudaMalloc(&output_device, n*sizeof(REAL));

    cudaMemcpy(data_device, input, n*sizeof(REAL), cudaMemcpyHostToDevice);

    // Perform axpy elements
    switch (kernel) {
    case 0: {
        while (n > 1) {
        //printf("Before Size: %d\n", n);
            global_1perThread<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(data_device, n);
            if (n%2 == 1) {
                n = n/2 + 1;
            }
            else {
                n /= 2;
            };
        };
        cudaMemcpy(&result, data_device, sizeof(REAL), cudaMemcpyDeviceToHost);
        break;
    }
    case 1: {
        global_block<<<BLOCK_NUM, BLOCK_SIZE>>>(data_device, output_device, n);
        int n2;
        if (n/BLOCK_NUM*BLOCK_SIZE == 0) {
            n2 = n;
        }
        else {
            n2 = n%(BLOCK_NUM*BLOCK_SIZE);
        };
        while (n2 > 1) {
            global_1perThread<<<(n2+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(output_device, n2);
            if (n2%2 == 1) {
                n2 = n2/2 + 1;
            }
            else {
                n2 /= 2;
            };
        };
        cudaMemcpy(&result, output_device, sizeof(REAL), cudaMemcpyDeviceToHost);
        break;
    }
    case 2: {
        global_cyclic<<<BLOCK_NUM, BLOCK_SIZE>>>(data_device, output_device, n);
        /*
        int n2;
        if (n/BLOCK_NUM*BLOCK_SIZE == 0) {
            n2 = n;
        }
        else {
            n2 = n%(BLOCK_NUM*BLOCK_SIZE);
        };
        while (n2 > 1) {
            global_1perThread<<<(n2+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(output_device, n2);
            if (n2%2 == 1) {
                n2 = n2/2 + 1;
            }
            else {
                n2 /= 2;
            };
        };
        */
        final_reduce(output_device, n);
        cudaMemcpy(&result, output_device, sizeof(REAL), cudaMemcpyDeviceToHost);
        break;
    }
    }
    cudaFree(data_device);
    cudaFree(output_device);
    return result;
}
