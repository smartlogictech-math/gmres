/**
 * @file vadd_demo.cc
 * @author zhe.zhang
 * @date 2025-03-24 11:29:58
 * @brief
 * @attention
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "vadd_api.h"

int main()
{
  const int N = 1024;
  std::vector<float> hA(N, 1.0f);
  std::vector<float> hB(N, 2.0f);
  std::vector<float> hC(N, 0.0f);

  float *dA, *dB, *dC;
  cudaMalloc(&dA, N * sizeof(float));
  cudaMalloc(&dB, N * sizeof(float));
  cudaMalloc(&dC, N * sizeof(float));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMemcpyAsync(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);

  int ret = vadd(dA, dB, dC, N, stream);
  if (0 != ret)
  {
    printf("Error: vadd return %d\n", ret);
  }
  else
  {
    cudaMemcpyAsync(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("Info: vadd completed\n");
  }

  cudaStreamDestroy(stream);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}