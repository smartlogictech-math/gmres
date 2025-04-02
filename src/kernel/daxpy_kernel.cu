/**
 * @file daxpy_kernel.cu
 * @author yu.xiao
 * @date 2025-03-28 19:05:46
 * @version 
 * @brief 
 * @attention
 */
#include "blas_handle.h"
#include "internal/hal/blas_daxpy_hal.h"
#include "internal/kernel/daxpy_kernel.h"


#define ELEMENTS_PER_THREAD 256 ///< 可调节
#define THREADS_PER_BLOCK 32    ///< 可调节    

__device__ blasStatus_t memcpyGlobalToLocal(double* dst, const double* src, const unsigned start, const unsigned n){
    
    for(int i = 0; i < n; i++){
        dst[i] = src[start+i];
    }

    return BLAS_STATUS_SUCCESS;
}

__device__ blasStatus_t memcpyLocalToGlobal(double* dst, const double* src, const unsigned start, const unsigned n){
    
    for(int i = 0; i < n; i++){
        dst[start+i] = src[i];
    }

    return BLAS_STATUS_SUCCESS;
}

__device__ blasStatus_t daxpyHal(const double* localX,const double* localY,double* localRet,const double alpha,const unsigned n){

    for(int i=0; i<n; i++){
        localRet[i] = alpha * localX[i] + localY[i];
    }

    return BLAS_STATUS_SUCCESS;
}

__global__ void daxpyKernel(unsigned n, double* alpha, const double* x, const double* y, double* result){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * ELEMENTS_PER_THREAD;
    int end = start + ELEMENTS_PER_THREAD;
    // 确保不超过数组边界
    if (end > n) {
        end = n;
    }
    unsigned cnt = end - start;
    double* localX = (double*)malloc(cnt * sizeof(double));
    double* localY = (double*)malloc(cnt * sizeof(double));
    double* localRet = (double*)malloc(cnt * sizeof(double));

    memcpyGlobalToLocal(localX,x,start,cnt);
    memcpyGlobalToLocal(localY,y,start,cnt);

    daxpyHal(localX,localY,localRet,*alpha,cnt);

    memcpyLocalToGlobal(result,localRet,start,cnt);

    
}
void launchDaxpy(blasHandle_t handle, unsigned n, double* alpha, const double* x, const double* y, double* result){

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n + THREADS_PER_BLOCK * ELEMENTS_PER_THREAD - 1) / (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD));

    cudaStream_t stream = handle->stream;
    daxpyKernel<<<grid, block, 0, stream>>>(n,alpha,x,y,result);

    cudaDeviceSynchronize();
}