/**
 * @file TEST_ddot_kernel.cc
 * @author zhe.zhang
 * @date 2025-03-31 13:42:39
 * @brief 
 * @attention 
 */

#include "ddot_kernel.h"

#include <gtest/gtest.h>

static void funcTest(const int n){
    blasHandle_t handle;
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasSetStream(handle, stream));

    double *dX, *dY;
    double *hX, *hY;

    cudaMalloc(&dX, n * sizeof(double));
    cudaMalloc(&dY, n * sizeof(double));
    cudaMallocHost(&hX, n * sizeof(double));
    cudaMallocHost(&hY, n * sizeof(double));

    for (int i = 0; i < n; i++){
        hX[i] = 2.0;
        hY[i] = 3.0;
    }

    double result;
    double *dResult;
    cudaMalloc(&dResult, sizeof(double));

    cudaMemcpyAsync(dX, hX, sizeof(double) * n, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dY, hY, sizeof(double) * n, cudaMemcpyHostToDevice, stream);

    launchDdot(handle, n, dX, dY, dResult);

    cudaMemcpyAsync(&result, dResult, sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    double expected_val = 2.0 * 3.0 * n;

    ASSERT_EQ(expected_val, result);

    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle));
    cudaStreamDestroy(stream);
    cudaFree(dX);
    cudaFree(dY);
    cudaFreeHost(hX);
    cudaFreeHost(hY);
    cudaFree(dResult);
}

TEST(BlasDdotKernelTest, SmallScaleTest){
    funcTest(7);
}

TEST(BlasDdotKernelTest, LargeScaleTest){
    funcTest((1<<21) + 7);
}