/**
 * @file blas_ddot_api.cc
 * @author zhe.zhang
 * @date 2025-03-30 10:21:43
 * @brief 
 * @attention 
 */

#include "blas_ddot_api.h"

#include "internal/hal/blasContext_impl.h"

static bool checkParas(blasHandle_t handle, const int n, const double *x, const double *y, const double *result){
    if((0 >= n) ||(nullptr == x) || (nullptr == y) || (nullptr == result)){
        handle->last_error = BLAS_STATUS_INVALID_VALUE;
        return false;
    }
 
    cudaPointerAttributes attr_x, attr_y;
    cudaPointerGetAttributes(&attr_x, x);
    cudaPointerGetAttributes(&attr_y, y);
    if(!((cudaMemoryTypeDevice == attr_x.type) && (cudaMemoryTypeDevice == attr_y.type))){
        handle->last_error = BLAS_STATUS_INVALID_VALUE;
        return false;
    }

    return true;
 }

extern "C" blasStatus_t blasDdot(blasHandle_t handle, const int n, const double *x, const double *y, double *result){
    if(!checkParas(handle, n, x, y, result)){
        return BLAS_STATUS_INVALID_VALUE;
    }
    
    return BLAS_STATUS_SUCCESS;
}