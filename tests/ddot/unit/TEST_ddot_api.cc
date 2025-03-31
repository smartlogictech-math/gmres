/**
 * @file TEST_ddot_api.cc
 * @author zhe.zhang
 * @date 2025-03-28 14:54:43
 * @brief 
 * @attention 
 */

#include "blas_ddot_api.h"

#include <gtest/gtest.h>

TEST(BlasDdotAPITest, InputParams){
    blasHandle_t handle;
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle));

    double *dX, *dY;
    double *hX, *hY;
    int n = 1;

    cudaMalloc(&dX, n * sizeof(double));
    cudaMalloc(&dY, n * sizeof(double));
    cudaMallocHost(&hX, n * sizeof(double));
    cudaMallocHost(&hY, n * sizeof(double));

    double result;
    double *dResult;
    cudaMalloc(&dResult, sizeof(double));

    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDdot(handle, n, dX, dY, &result));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDdot(handle, n, dX, dY, dResult));

    /// check n
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDdot(handle, 0, dX, dY, &result));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDdot(handle, -1, dX, dY, &result));

    /// check ptr
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDdot(handle, n, nullptr, dY, &result));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDdot(handle, n, dX, nullptr, &result));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDdot(handle, n, dX, dY, nullptr));
    
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDdot(handle, n, hX, dY, &result));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDdot(handle, n, dX, hY, &result));

    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle));
    
    cudaFree(dX);
    cudaFree(dY);
    cudaFreeHost(hX);
    cudaFreeHost(hY);
    cudaFree(dResult);
}

TEST(BlasDdotAPITest, ResultAddr){
    const int n = (1 << 21) + 7;
    blasHandle_t handle0, handle1;
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle0));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle1));

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasSetStream(handle0, stream0));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasSetStream(handle1, stream1));

    double *dX, *dY;
    double *hX, *hY;

    cudaMalloc(&dX, n * sizeof(double));
    cudaMalloc(&dY, n * sizeof(double));
    cudaMallocHost(&hX, n * sizeof(double));
    cudaMallocHost(&hY, n * sizeof(double));

    double *dResult;
    cudaMallocAsync(&dResult, sizeof(double), stream1);

    for (int i = 0; i < n; i++){
        hX[i] = 2.0;
        hY[i] = 3.0;
    }

    double result0, result1;

    cudaMemcpy(dX, hX, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, sizeof(double) * n, cudaMemcpyHostToDevice);

    blasDdot(handle0, n, dX, dY, &result0);

    blasDdot(handle1, n, dX, dY, dResult);
    cudaMemcpyAsync(&result1, dResult, sizeof(double), cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    double expected_val = 2.0 * 3.0 * n;

    ASSERT_EQ(expected_val, result0);
    ASSERT_EQ(expected_val, result1);

    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle0));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle1));
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaFree(dX);
    cudaFree(dY);
    cudaFreeHost(hX);
    cudaFreeHost(hY);
    cudaFree(dResult);
}