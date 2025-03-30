/**
 * @file TEST_ddot_api.cc
 * @author zhe.zhang
 * @date 2025-03-28 14:54:43
 * @brief 
 * @attention 
 */

#include "blas_ddot_api.h"

#include "fixtures/ddot_api_fixture.h"

TEST_F(BlasDdotAPITest, input_params){
    ASSERT_NE(SetVecLen(1), nullptr);
    Malloc();

    double result;
    double *dResult;
    ASSERT_EQ(cudaMalloc(&dResult, sizeof(double)), cudaSuccess);

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
}