/**
 * @file TEST_vadd_kernel.cc
 * @author zhe.zhang
 * @date 2025-03-23 17:48:17
 * @brief 
 * @attention 
 */

#include "internal/kernel/vadd_kernel.h"

#include <gtest/gtest.h>

TEST(VaddKernelTest, BasicAddition) {
  const int n = 3;
  float hA[3] = {1.0f, 2.0f, 3.0f};
  float hB[3] = {4.0f, 5.0f, 6.0f};
  float hC[3] = {0};

  float *dA, *dB, *dC;
  cudaError_t err;

  err = cudaMalloc(&dA, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&dB, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&dC, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  err = cudaMemcpyAsync(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMemcpyAsync(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);

  launchVadd(dA, dB, dC, n, 0);

  err = cudaMemcpyAsync(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  ASSERT_EQ(err, cudaSuccess);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  EXPECT_FLOAT_EQ(hC[0], 5.0f);
  EXPECT_FLOAT_EQ(hC[1], 7.0f);
  EXPECT_FLOAT_EQ(hC[2], 9.0f);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

TEST(VaddKernelTest, LargeScaleAddition) {
  const int n = 1 << 20;
  std::vector<float> hA(n, 1.0f);
  std::vector<float> hB(n, 2.0f);
  std::vector<float> hC(n, 0.0f);

  float *dA, *dB, *dC;
  cudaError_t err;

  err = cudaMalloc(&dA, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&dB, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMalloc(&dC, n * sizeof(float));
  ASSERT_EQ(err, cudaSuccess);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  err = cudaMemcpyAsync(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);
  err = cudaMemcpyAsync(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);

  launchVadd(dA, dB, dC, n, stream);

  err = cudaMemcpyAsync(hC.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  ASSERT_EQ(err, cudaSuccess);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(hC[i], 3.0f, 1e-5);
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
