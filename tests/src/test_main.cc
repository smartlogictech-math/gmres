/**
 * @file test_main.cc
 * @author zhe.zhang
 * @date 2025-03-18 15:08:46
 * @brief 
 * @attention 
 */

#include "gtest/gtest.h"
#include <iostream>

class GlobalTestEnvironment : public ::testing::Environment {
    public:
        /// @brief Executed before all test cases
        void SetUp() override {
            std::cout << "===== Global Test Start =====" << std::endl;
        }
        /// @brief Executed after all test cases
        void TearDown() override {
            std::cout << "===== Global Test End =====" << std::endl;
        }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    // Register the global test environment, gtest manages its lifecycle
    ::testing::AddGlobalTestEnvironment(new GlobalTestEnvironment);
    
    return RUN_ALL_TESTS();
}