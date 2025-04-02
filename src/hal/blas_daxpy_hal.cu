/**
 * @file blas_daxpy_hal.cu
 * @author yu.xiao
 * @date 2025-04-01 14:34:55
 * @version 
 * @brief 
 * @attention
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include "internal/hal/blas_daxpy_hal.h"

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
