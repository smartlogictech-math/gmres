/**
 * @file blas_ddot_api.cc
 * @author zhe.zhang
 * @date 2025-03-30 10:21:43
 * @brief 
 * @attention 
 */

#include "blas_ddot_api.h"

extern "C" blasStatus_t blasDdot(blasHandle_t handle, const int n, const double *x, const double *y, double *result){
    return BLAS_STATUS_SUCCESS;
}