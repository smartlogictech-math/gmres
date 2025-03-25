/**
 * @file TEST_vadd_api.cc
 * @author zhe.zhang
 * @date 2025-03-23 21:15:16
 * @brief 
 * @attention 
 */

#include "vadd_api.h"

#include "fixtures/vadd_api_fixture.h"

TEST_F(VaddAPITest, SmallScaleTest) {
    const int n = 5;
    float hA[n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float hB[n] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float hC[n] = {0};

    float *dA, *dB, *dC;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMalloc(&dC, n * sizeof(float));

    cudaMemcpyAsync(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice, stream);

    int ret = vadd(dA, dB, dC, n, stream);
    ASSERT_EQ(0, ret);
    cudaMemcpyAsync(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    EXPECT_FLOAT_EQ(hC[0], 6.0f);
    EXPECT_FLOAT_EQ(hC[1], 6.0f);
    EXPECT_FLOAT_EQ(hC[2], 6.0f);
    EXPECT_FLOAT_EQ(hC[3], 6.0f);
    EXPECT_FLOAT_EQ(hC[4], 6.0f);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

TEST_F(VaddAPITest, LargeScaleTest) {
    const int n = 1 << 20;
    std::vector<float> hA(n, 1.0f);
    std::vector<float> hB(n, 2.0f);
    std::vector<float> hC(n, 0.0f);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMalloc(&dC, n * sizeof(float));

    cudaMemcpyAsync(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream);

    int ret = vadd(dA, dB, dC, n, stream);
    ASSERT_EQ(0, ret);
    cudaMemcpyAsync(hC.data(), dC, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < n; i += n / 100) {
        EXPECT_NEAR(hC[i], 3.0f, 1e-5);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

TEST_F(VaddAPITest, InvalidInput) {
    float *dC;
    cudaMalloc(&dC, sizeof(float));

    EXPECT_EQ(vadd(nullptr, nullptr, dC, 1, stream), -1);
    EXPECT_EQ(vadd(nullptr, nullptr, nullptr, 1, stream), -1);
    EXPECT_EQ(vadd(nullptr, nullptr, nullptr, -1, stream), -1);

    float *hA, *hB, *hC;
    cudaMallocHost(&hA, sizeof(float));
    cudaMallocHost(&hB, sizeof(float));
    cudaMallocHost(&hC, sizeof(float));

    EXPECT_EQ(vadd(hA, hB, hC, 1, stream), -1);

    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
}
