/**
 * @file blas_handle.h
 * @author zhe.zhang
 * @date 2025-03-27 20:56:03
 * @brief 
 * @attention 
 */
#ifndef _BLAS_HANDLE_H_
#define _BLAS_HANDLE_H_

#include <cuda_runtime.h>

#include "blas_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct blasContext* blasHandle_t;

blasStatus_t blasCreate(blasHandle_t* handle);
blasStatus_t blasDestroy(blasHandle_t handle);

blasStatus_t blasSetStream(blasHandle_t handle, cudaStream_t stream);
blasStatus_t blasGetStream(blasHandle_t handle, cudaStream_t *stream);

#ifdef __cplusplus
}
#endif

#endif // _BLAS_HANDLE_H_