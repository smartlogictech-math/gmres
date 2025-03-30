/**
 * @file ddot_api_fixture.cc
 * @author zhe.zhang
 * @date 2025-03-30 09:05:00
 * @brief 
 * @attention 
 */

#include "ddot_api_fixture.h"

#include <iostream>

void BlasDdotAPITest::SetUpTestSuite(){
    std::cout << "=== Test Suite BlasDdotAPI Setup ===" << std::endl;
}

void BlasDdotAPITest::TearDownTestSuite(){
    std::cout << "=== Test Suite BlasDdotAPI Teardown ===" << std::endl;
}

void BlasDdotAPITest::SetUp(){
    ASSERT_EQ(blasCreate(&handle), BLAS_STATUS_SUCCESS);
    ASSERT_EQ(cudaStreamCreate(&stream), BLAS_STATUS_SUCCESS);
    ASSERT_EQ(blasSetStream(handle, stream), BLAS_STATUS_SUCCESS);
    dX = nullptr;
    dY = nullptr;
    hX = nullptr;
    hY = nullptr;
}

void BlasDdotAPITest::TearDown(){
    ASSERT_EQ(blasDestroy(handle), BLAS_STATUS_SUCCESS);
    ASSERT_EQ(cudaStreamDestroy(stream), BLAS_STATUS_SUCCESS);
    cudaFree(dX);
    cudaFree(dY);
    cudaFreeHost(hX);
    cudaFreeHost(hY);
}

BlasDdotAPITest* BlasDdotAPITest::SetVecLen(const int len){
    if(len <= 0){
        n = 0;
        return nullptr;
    }else{
        n = len;
        return this;
    }
}

void BlasDdotAPITest::Malloc(){
    ASSERT_EQ(cudaMalloc(&dX, n * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dY, n * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMallocHost(&hX, n * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMallocHost(&hY, n * sizeof(double)), cudaSuccess);
}