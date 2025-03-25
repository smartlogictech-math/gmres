/**
 * @file vadd_api_fixture.cc
 * @author zhe.zhang
 * @date 2025-03-23 21:20:44
 * @brief 
 * @attention 
 */

#include "fixtures/vadd_api_fixture.h"

#include <iostream>

void VaddAPITest::SetUpTestSuite(){
    std::cout << "=== Test Suite VaddAPI Setup ===" << std::endl;
}

void VaddAPITest::TearDownTestSuite(){
    std::cout << "=== Test Suite VaddAPI Teardown ===" << std::endl;
}

void VaddAPITest::SetUp(){
    std::cout << "[Test Setup]" << std::endl;
    cudaStreamCreate(&stream);
}

void VaddAPITest::TearDown(){
    cudaStreamDestroy(stream);
    std::cout << "[Test Teardown]" << std::endl;
}