/**
 * @file daxpy_kernel.h
 * @author yu.xiao
 * @date 2025-04-01 15:43:01
 * @version 
 * @brief 
 * @attention
 */

#ifndef DAXPY_KERNEL_H
#define DAXPY_KERNEL_H

#include "blas_handle.h"

#ifdef __cplusplus
extern "C" {
#endif


__global__ void daxpyKernel(unsigned n, double* alpha, const double* x, const double* y, double* result);

void launchDaxpy(blasHandle_t handle, unsigned n, double* alpha, const double* x, const double* y, double* result);

#ifdef __cplusplus
}
#endif

#endif // DAXPY_KERNEL_H