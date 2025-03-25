/**
 * @file vadd_api.h
 * @author zhe.zhang
 * @date 2025-03-23 17:14:34
 * @brief 
 * @attention 
 */
#pragma once

#include <cuda_runtime.h>

/**
 * @brief vector add, c=a+b
 * 
 * @param dA address of a in device memory
 * @param dB address of b in device memory
 * @param dC address of c in device memory
 * @param n length of vector
 * @param stream 
 * @return int 
 * @retval 0: success
 * @retval -1: failure
 */
int vadd(const float* dA, const float* dB, float* dC, int n, cudaStream_t stream);