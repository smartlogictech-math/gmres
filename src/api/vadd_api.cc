/**
 * @file vadd_api.cc
 * @author zhe.zhang
 * @date 2025-03-23 17:34:31
 * @brief 
 * @attention 
 */

#include "vadd_api.h"
#include "internal/kernel/vadd_kernel.h"

#include <cstdio>

static bool checkParas(const float* const dA, const float* const dB, const float* const dC, const int n){
   if((nullptr == dA) || (nullptr == dB) || (nullptr == dC) || (0 >= n)){
      fprintf(stderr, "%s(%u): Invalid arguments: dA=%p, dB=%p, dC=%p, n=%d\n", __FUNCTION__, __LINE__, dA, dB, dC, n);
      fflush(stderr);
      return false;
   }

   cudaPointerAttributes attr_a, attr_b, attr_c;
   cudaPointerGetAttributes(&attr_a, dA);
   cudaPointerGetAttributes(&attr_b, dB);
   cudaPointerGetAttributes(&attr_c, dC);
   if(!((cudaMemoryTypeDevice == attr_a.type) && (cudaMemoryTypeDevice == attr_b.type) && (cudaMemoryTypeDevice == attr_c.type))){
      fprintf(stderr, "%s(%u): Invalid memory type: attr_a.type=%d, attr_b.type=%d, attr_c.type=%d\n",
              __FUNCTION__, __LINE__, attr_a.type, attr_b.type, attr_c.type);
      fflush(stderr);
      return false;
   }

   return true;
}

int vadd(const float* dA, const float* dB, float* dC, int n, cudaStream_t stream) {
   /// If considering performance overhead, comment out the parameter checking function
   if(!checkParas(dA, dB, dC, n)){
      return -1;
   }
 
   launchVadd(dA, dB, dC, n, stream);

   return 0;
}