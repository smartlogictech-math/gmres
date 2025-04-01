/**
 * @file reduce_1Block_kernel.cu
 * @author zhe.zhang
 * @date 2025-03-31 15:42:46
 * @brief 
 * @attention 
 */

#include "blas_handle.h"

#include "internal/hal/blasContext_impl.h"

#define THREADS_PER_BLOCK 1024

__global__ void reduce_1Block_kernel(const double* input, double* output, int size) {
    extern __shared__ double sharedMem[];
    int tid = threadIdx.x;
    int idx = threadIdx.x;
    double sum = 0.0;

    for (int i = idx; i < size; i += blockDim.x) {
        sum += input[i];
    }
    sharedMem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sharedMem[0];
    }
}

extern "C" void launchDReduce1Block(blasHandle_t handle, const int n, const double *vector, double *sum){
    reduce_1Block_kernel<<<1, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double), handle->stream>>>(vector, sum, n);
}