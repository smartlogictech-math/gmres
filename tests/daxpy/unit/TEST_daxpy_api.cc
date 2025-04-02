/**
 * @file TEST_daxpy_api.cc
 * @author yu.xiao
 * @date 2025-04-02 16:10:01
 * @version 
 * @brief 
 * @attention
 */

#include <gtest/gtest.h>
#include "blas_daxpy_api.h"


static bool cmpData(const double* a, const double num, const unsigned n){
    for(unsigned i=0;i<n;i++){
        if(num != a[i]){
            return false;
        }
    }
    return true;
}

TEST(BlasDaxpyAPITest,InputParams){
    /// handle
    blasHandle_t handle;
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle));

    double *dX, *dY, *dResult, *dAlpha;
    double *hX, *hY, *hResult, alpha;
    int n = 1;

    cudaMalloc(&dX, n * sizeof(double));
    cudaMalloc(&dY, n * sizeof(double));
    cudaMalloc(&dResult, n * sizeof(double));
    cudaMalloc(&dAlpha, sizeof(double));

    hX = (double*)malloc(n * sizeof(double));
    hY = (double*)malloc(n * sizeof(double));
    hResult = (double*)malloc(n * sizeof(double));

    /// alpha
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDaxpy(handle, n, dAlpha, dX, dY, dResult));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDaxpy(handle, n, &alpha, dX, dY, dResult));
    
    /// n
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, 0, &alpha, dX, dY, dResult));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, -1, &alpha, dX, dY, dResult));
    
    /// ptr
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, n, dAlpha, nullptr, dY, dResult));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, n, dAlpha, dX, nullptr, dResult));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, n, dAlpha, dX, dY, nullptr));

    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, n, dAlpha, hX, dY, dResult));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, n, dAlpha, dX, hY, dResult));
    ASSERT_EQ(BLAS_STATUS_INVALID_VALUE, blasDaxpy(handle, n, dAlpha, dX, dY, hResult));
   
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle));

    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dResult);
    cudaFree(dAlpha);

    free(hX);
    free(hY);
    free(hResult);
}

TEST(BlasDaxpyAPITest, AlphaAddr){

    blasHandle_t handle0, handle1;
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle0));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle1));

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasSetStream(handle0, stream0));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasSetStream(handle1, stream1));

    double *dX, *dY, *dResult, *dAlpha;
    double *hX, *hY, *hResult1, *hResult2, alpha = 2.0;
    const int n = (1 << 21) + 7;

    cudaMalloc(&dX, n * sizeof(double));
    cudaMalloc(&dY, n * sizeof(double));
    cudaMalloc(&dResult, n * sizeof(double));
    cudaMalloc(&dAlpha, sizeof(double));
    *dAlpha = 2.0;

    cudaMallocHost(&hX, n * sizeof(double));
    cudaMallocHost(&hY, n * sizeof(double));
    cudaMallocHost(&hResult1, n * sizeof(double));
    cudaMallocHost(&hResult2, n * sizeof(double));


    for(int i = 0; i < n; i++){
        hX[i] = 3.0;
        hY[i] = 1.0;
    }

    cudaMemcpy(dX, hX, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, sizeof(double) * n, cudaMemcpyHostToDevice);

    blasDaxpy(handle0,  n, &alpha, dX, dY, dResult);
    blasDaxpy(handle1,  n, dAlpha, dX, dY, dResult);

    cudaMemcpyAsync(hResult1, dResult, n*sizeof(double), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(hResult2, dResult, n*sizeof(double), cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    ASSERT_EQ(true,cmpData(hResult1,5,n));
    ASSERT_EQ(true,cmpData(hResult2,5,n));


    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle0));
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle1));
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dResult);
    cudaFree(dAlpha);

    cudaFreeHost(hX);
    cudaFreeHost(hY);
    cudaFreeHost(hResult1);
    cudaFreeHost(hResult2);


}