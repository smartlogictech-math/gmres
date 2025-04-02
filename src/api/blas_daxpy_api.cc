/**
 * @file blas_daxpy_api.cc
 * @author yu.xiao
 * @date 2025-03-28 17:22:24
 * @version 
 * @brief 
 * @attention
 */
#include "blas_daxpy_api.h"
#include "internal/kernel/daxpy_kernel.h"
#include <stdio.h>


static bool checkParas(blasHandle_t handle, const double* alpha, const double* x, const double* y, const double* ret, int n){
    if(nullptr == alpha || nullptr == x || nullptr == y || nullptr == ret || 0 >= n){
        fprintf(stderr, "%s(%u): Invalid arguments: x=%p, y=%p, ret=%p, n=%d\n", __FUNCTION__, __LINE__, x, y, ret, n);
        fflush(stderr);
        handle->last_error = BLAS_STATUS_INVALID_VALUE;
        return false;
    }

    cudaPointerAttributes attrX, attrY, attrRet;
    cudaPointerGetAttributes(&attrX, x);
    cudaPointerGetAttributes(&attrY, y);
    cudaPointerGetAttributes(&attrRet, ret);
    if(!((cudaMemoryTypeDevice == attrX.type) && (cudaMemoryTypeDevice == attrY.type) && (cudaMemoryTypeDevice == attrRet.type))){
      fprintf(stderr, "%s(%u): Invalid memory type: attrX.type=%d, attrY.type=%d, attrRet.type=%d\n",
              __FUNCTION__, __LINE__, attrRet.type, attrY.type, attrRet.type);
      fflush(stderr);
      handle->last_error = BLAS_STATUS_INVALID_VALUE;
      return false;
    }

    return true;
}

blasStatus_t blasDaxpy(blasHandle_t handle, unsigned n, double* alpha, const double* x, const double* y, double* result){

    if(!checkParas(handle, alpha, x, y, result, n)){
        return BLAS_STATUS_INVALID_VALUE;
    }

    cudaPointerAttributes attrAlpha;
    cudaPointerGetAttributes(&attrAlpha, alpha);

    double* dAlpha;
    if(cudaMemoryTypeDevice == attrAlpha.type){
        dAlpha = alpha;
    }else{
        cudaMalloc(&dAlpha, sizeof(double));
    }

    launchDaxpy(handle, n, dAlpha, x, y, result);

    return BLAS_STATUS_SUCCESS;
}