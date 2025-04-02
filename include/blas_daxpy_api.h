/**
 * @file blas_daxpy_api.h
 * @author yu.xiao
 * @date 2025-04-01 15:51:47
 * @version 
 * @brief 
 * @attention
 */

#ifndef BLAS_DAXPY_API_H
#define BLAS_DAXPY_API_H

#include "blas_handle.h"
#include "internal/hal/blasContext_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief result = alpha*x + y
 * 
 * @param handle handle to the cuBLAS library context.
 * @param n number of elements in the vectors x and y.
 * @param alpha host or device, scalar used for multiplication.
 * @param x device, vector with n elements.
 * @param y device, vector with n elements.
 * @param result device, the result vector with n elements.
 * @return blasStatus_t 
 */
blasStatus_t blasDaxpy(blasHandle_t handle, unsigned n, double* alpha, const double* x, const double* y, double* result);


#ifdef __cplusplus
}
#endif

#endif // BLAS_DAXPY_API_H