/**
 * @file blasContext_impl.h
 * @author zhe.zhang
 * @date 2025-03-28 17:19:03
 * @brief 
 * @attention 
 */
#ifndef _BLASCONTEXT_IMPL_H_
#define _BLASCONTEXT_IMPL_H_

#include <cuda_runtime.h>

#include "blas_status.h"

#ifdef __cplusplus
extern "C" {
#endif

struct blasContext{
    cudaStream_t stream;
    blasStatus_t last_error;
    void *workspace;
    size_t workspace_sz;
};

#ifdef __cplusplus
}
#endif

#endif // _BLASCONTEXT_IMPL_H_