/**
 * @file ddot_kernel.h
 * @author zhe.zhang
 * @date 2025-03-27 21:02:02
 * @brief 
 * @attention 
 */
#ifndef _DDOT_KERNEL_H_
#define _DDOT_KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "blas_handle.h"

void launchDdot(blasHandle_t handle, const int n, const double *x, const double *y, const double *result);
__global__ void dotDKernel(const int N, const double *x, const double *y, const double *result);

#ifdef __cplusplus
}
#endif

#endif // _DDOT_KERNEL_H_
