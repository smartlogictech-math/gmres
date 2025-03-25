/**
 * @file vadd_api_fixture.h
 * @author zhe.zhang
 * @date 2025-03-23 21:20:27
 * @brief 
 * @attention 
 */

#pragma once

#include <cuda_runtime.h>

#include <gtest/gtest.h>

// Define a test fixture class
class VaddAPITest: public ::testing::Test {
    protected:
        // Runs once before any test in this test suite
        static void SetUpTestSuite();
        // Runs once after all tests in this test suite have run
        static void TearDownTestSuite();

        // Runs before each test case
        void SetUp() override;
        // Runs after each test case
        void TearDown() override;
    
        cudaStream_t stream;
};