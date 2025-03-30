/**
 * @file blasContext_impl.cu
 * @author zhe.zhang
 * @date 2025-03-28 17:23:21
 * @brief 
 * @attention 
 */

#include "internal/hal/blasContext_impl.h"
#include "blas_handle.h"

extern "C" blasStatus_t blasCreate(blasHandle_t* handle) {
    if (!handle){
        return BLAS_STATUS_INVALID_VALUE;
    }

    struct blasContext* h = new struct blasContext;
    h->last_error = BLAS_STATUS_SUCCESS;
    h->stream = 0;

    h->workspace_sz = 1024 * 1024; // 1MB
    cudaError_t  err = cudaMalloc(&h->workspace, h->workspace_sz);
    if (err != cudaSuccess) {
        delete h;
        return BLAS_STATUS_ALLOC_FAILED;
    }

    *handle = h;
    return BLAS_STATUS_SUCCESS;
}

extern "C" blasStatus_t blasDestroy(blasHandle_t handle) {
    if (!handle){
        return BLAS_STATUS_INVALID_VALUE;
    }

    cudaFree(handle->workspace);
    delete handle;
    return BLAS_STATUS_SUCCESS;
}

extern "C" blasStatus_t blasSetStream(blasHandle_t handle, cudaStream_t stream) {
    if (!handle){
        return BLAS_STATUS_NOT_INITIALIZED;
    }
    handle->stream = stream;
    return BLAS_STATUS_SUCCESS;
}

extern "C" blasStatus_t blasGetStream(blasHandle_t handle, cudaStream_t *stream){
    if (!handle || !stream) {
        return BLAS_STATUS_INVALID_VALUE;
    }
    *stream = handle->stream;
    return BLAS_STATUS_SUCCESS;
}