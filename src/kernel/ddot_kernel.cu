/**
 * @file ddot_kernel.cu
 * @author zhe.zhang
 * @date 2025-03-31 13:46:57
 * @brief 
 * @attention 
 */

#include "blas_handle.h"
#include "internal/hal/blasContext_impl.h"

__global__ void ddotKernel(const int n, const double *x, const double *y, double *tmpResult, const int N){
    extern __shared__ double ddotRet[];
    unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int dataIdx = threadId * N;
    
    double sum = 0.0;

    for (int i = 0; i < N; i++){
        if(dataIdx < n){
            sum += x[dataIdx] * y[dataIdx];
        }else{
            break;
        }
        dataIdx++;
    }
    ddotRet[threadIdx.x] = sum;
    __syncthreads();
    int validThreadNum = (n + N - 1) / N;
    if(threadId < validThreadNum){
        tmpResult[threadId] = ddotRet[threadIdx.x];
    }
}

static int getThreadProcLen(){
    /// @todo set the length under different architectures
    const int Nvidia_4090_len = 16;
    return Nvidia_4090_len;
}

static uint getBlockDimx(){
    /// @todo set the dim.x under different architectures
    return 32;
}

extern "C" void launchDReduce1Block(blasHandle_t handle, const int n, const double *vector, double *sum);

extern "C" void launchDdot(blasHandle_t handle, const int n, const double *x, const double *y, double *result){
    int N = getThreadProcLen();
    int validThreadNum = (n + N - 1) / N;
    dim3 block(getBlockDimx(), 1, 1);
    dim3 grid((validThreadNum + block.x - 1) / block.x, 1, 1);
    uint shmSz = block.x * sizeof(double);

    cudaStream_t stream = handle->stream;

    bool tmpRetSpaceAllocated = (handle->workspace_sz > validThreadNum * sizeof(double));
    double *tmpResult = nullptr;
    if (!tmpRetSpaceAllocated){
        cudaMallocAsync(&tmpResult, validThreadNum * sizeof(double), stream);
    }else{
        tmpResult = (double *)handle->workspace;
    }

    ddotKernel<<<grid, block, shmSz, stream>>>(n, x, y, tmpResult, N);

    launchDReduce1Block(handle, validThreadNum, tmpResult, result);

    if(!tmpRetSpaceAllocated){
        cudaFreeAsync(tmpResult, stream);
    }
}