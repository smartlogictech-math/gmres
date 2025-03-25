/**
 * @file vadd_kernel.h
 * @author zhe.zhang
 * @date 2025-03-23 17:32:24
 * @brief 
 * @attention 
 */
#pragma once

#include <cuda_runtime.h>

__global__ void vadd_kernel(const float* a, const float* b, float* c, int n);
void launchVadd(const float *dA, const float *dB, float *dC, int n, cudaStream_t stream);