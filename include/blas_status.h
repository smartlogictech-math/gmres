/**
 * @file blas_status.h
 * @author zhe.zhang
 * @date 2025-03-28 17:34:14
 * @brief 
 * @attention 
 */
#ifndef _BLAS_STATUS_H_
#define _BLAS_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

    typedef enum
    {
        BLAS_STATUS_SUCCESS = 0,
        BLAS_STATUS_NOT_INITIALIZED = 1,
        BLAS_STATUS_ALLOC_FAILED = 2,
        BLAS_STATUS_INVALID_VALUE = 3,
        BLAS_STATUS_INTERNAL_ERROR = 4
    } blasStatus_t;

#ifdef __cplusplus
}
#endif

#endif // _BLAS_STATUS_H_