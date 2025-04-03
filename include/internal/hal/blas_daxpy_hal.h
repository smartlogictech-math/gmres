/**
 * @file blas_daxpy_hal.h
 * @author yu.xiao
 * @date 2025-04-01 16:05:37
 * @version 
 * @brief 
 * @attention
 */

#ifndef BLAS_DAXPY_HAL_H
#define BLAS_DAXPY_HAL_H

#include "blas_handle.h"
#include "internal/hal/blasContext_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

__device__ blasStatus_t memcpyGlobalToLocal(double* dst, const double* src, const unsigned start, const unsigned n);

__device__ blasStatus_t memcpyLocalToGlobal(double* dst, const double* src, const int start, const int n);

__device__ blasStatus_t daxpyHal(const double* localX,const double* localY,double* localRet,const double alpha,const int n);




#ifdef __cplusplus
}
#endif

#endif // BLAS_DAXPY_HAL_H