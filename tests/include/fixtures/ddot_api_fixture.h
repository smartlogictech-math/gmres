/**
 * @file ddot_api_fixture.h
 * @author zhe.zhang
 * @date 2025-03-28 19:16:36
 * @brief 
 * @attention 
 */

#pragma once

#include "blas_handle.h"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

class BlasDdotAPITest: public ::testing::Test {
    protected:
        static void SetUpTestSuite();
        static void TearDownTestSuite();

        void SetUp() override;
        void TearDown() override;

        BlasDdotAPITest* SetVecLen(const int len);
        void Malloc();

        blasHandle_t handle;
        cudaStream_t stream;
        double *dX, *dY;
        double *hX, *hY;
        int n;
};