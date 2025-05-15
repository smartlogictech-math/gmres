/**
 * @file reduce_1Block_kernel.cu
 * @author zhe.zhang
 * @date 2025-03-31 15:42:46
 * @brief 
 * @attention 
 */

#include "blas_handle.h"

#include "internal/hal/blasContext_impl.h"
// #include "internal/hal/hal.h"

// #define MAX_BLOCKS_NUM  ()
#define MAX_THREADS_NUM (1024)
#define THREAD_PACKED_LEN (8)

// extern __device__ void (*NVIDIA_memcpy_async)(void*, const void*, size_t);

// __device__ void NVIDIA_memcpy_async(void *dst, const void *src, size_t size){
//     char* d = static_cast<char*>(dst);
//     const char* s = static_cast<const char*>(src);
    
//     for (size_t i = 0; i < size; i++) {
//         d[i] = s[i];
//     }
// }

__global__ void reduce_1Block_kernel(const double* input, double* output, int size) {
    extern __shared__ double sharedMem[];
    double packedData[THREAD_PACKED_LEN];
    int tid = threadIdx.x;
    int idx = threadIdx.x;
    double sum = 0.0;

    for (int i = idx * THREAD_PACKED_LEN; i < size; i += blockDim.x * THREAD_PACKED_LEN) {
        size_t validPackedLen = (i + THREAD_PACKED_LEN) > size ? (size - i) : THREAD_PACKED_LEN;
#if 0
        NVIDIA_memcpy_async(&packedData[0], &input[i], validPackedLen * sizeof(double));
        for (size_t dId = 0; dId < validPackedLen;++dId){
            packedData[dId] = input[i + dId];
        }
            // arrive_and_wait();

        for (size_t j = 0; j < validPackedLen; ++j)
        {
            sum += packedData[j];
        }
#endif
        for (size_t j = 0; j < validPackedLen; j++){
            sum += input[i + j];
        }
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

__global__ void reduce_kernel(const double* input, double* output, int size) {
    extern __shared__ double sharedMem[];
    double packedData[THREAD_PACKED_LEN];
    const unsigned int TotalThreadNum = blockDim.x * gridDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0.0;

    for (int i = tid * THREAD_PACKED_LEN; i < size; i += TotalThreadNum * THREAD_PACKED_LEN) {
        size_t validPackedLen = (i + THREAD_PACKED_LEN) > size ? (size - i) : THREAD_PACKED_LEN;
#if 0
        NVIDIA_memcpy_async(&packedData[0], &input[i], validPackedLen * sizeof(double));
        for (size_t dId = 0; dId < validPackedLen;++dId){
            packedData[dId] = input[i + dId];
        }
            arrive_and_wait();

        for (size_t j = 0; j < validPackedLen; ++j)
        {
            sum += packedData[j];
        }
#endif
        for (size_t j = 0; j < validPackedLen; j++){
            sum += input[i + j];
        }
    }
    sharedMem[threadIdx.x] = sum;
    __syncthreads();

    output[tid] = sharedMem[threadIdx.x];
}

extern "C" void launchDReduce1Block(blasHandle_t handle, const int n, const double *vector, double *sum){
    uint threadNumPerBlock = MAX_THREADS_NUM;
    uint blockNum = 256;
    double *tmpReduceRet;
    cudaMallocAsync(&tmpReduceRet, sizeof(double) * threadNumPerBlock * blockNum, handle->stream);
    reduce_kernel<<<blockNum, threadNumPerBlock, threadNumPerBlock * sizeof(double), handle->stream>>>(vector, tmpReduceRet, n);
    reduce_1Block_kernel<<<1, MAX_THREADS_NUM, MAX_THREADS_NUM * sizeof(double), handle->stream>>>(tmpReduceRet, sum, threadNumPerBlock * blockNum);
    cudaFreeAsync(tmpReduceRet, handle->stream);
}