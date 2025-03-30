/**
 * @file blas_ddot_api.h
 * @author zhe.zhang
 * @date 2025-03-27 20:50:12
 * @brief 
 * @attention 
 */

#ifndef _BLAS_DDOT_API_H_
#define _BLAS_DDOT_API_H_

#include "blas_handle.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief result = x' * y
 * 
 * @param handle handle to the cuBLAS library context.
 * @param n number of elements in the vectors x and y.
 * @param x device, vector with n elements.
 * @param y device, vector with n elements.
 * @param result host or device, the resulting dot product, which is 0.0 if n<=0.
 * @return blasStatus_t 
 */
blasStatus_t blasDdot(blasHandle_t handle, const int n, const double *x, const double *y, double *result);


#ifdef __cplusplus
}
#endif

#endif // _BLAS_DDOT_API_H_