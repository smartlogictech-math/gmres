/**
 * @file blas_handle.h
 * @author zhe.zhang
 * @date 2025-03-27 20:56:03
 * @brief 
 * @attention 
 */
#ifndef _BLAS_HANDLE_H_
#define _BLAS_HANDLE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>

struct blasContext{
    cudaStream_t stream;
};
typedef struct blasContext *blasHandle_t;

int blasSetStream(blasHandle_t handle, cudaStream_t stream){
    handle->stream = stream;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif // _BLAS_HANDLE_H_