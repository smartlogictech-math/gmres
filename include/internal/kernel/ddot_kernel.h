/**
 * @file ddot_kernel.h
 * @author zhe.zhang
 * @date 2025-03-27 21:02:02
 * @brief 
 * @attention 
 */
#ifndef _DDOT_KERNEL_H_
#define _DDOT_KERNEL_H_

#include "blas_handle.h"

#ifdef __cplusplus
extern "C" {
#endif

    __global__ void ddotKernel(const int n, const double *x, const double *y, double *tmpResult, const int N);
    void launchDdot(blasHandle_t handle, const int n, const double *x, const double *y, double *result);

#ifdef __cplusplus
}
#endif

#endif // _DDOT_KERNEL_H_
