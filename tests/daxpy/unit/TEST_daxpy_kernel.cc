/**
 * @file TEST_daxpy_kernel.cc
 * @author yu.xiao
 * @date 2025-04-02 20:07:49
 * @version 
 * @brief 
 * @attention
 */

#include <gtest/gtest.h>
#include "daxpy_kernel.h"


static bool cmpData(const double* a, const double* b, const int n){
    for(int i=0;i<n;i++){
        if(a[i] != b[i]){
            return false;
        }
    }
    return true;
}

static double* read_vector_from_file(const char* filename, unsigned* n) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }
    
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    *n = file_size / sizeof(double);
    double* data = (double*)malloc(file_size);
    if (!data) {
        fclose(file);
        return NULL;
    }
    
    if (fread(data, sizeof(double), *n, file) != *n) {
        perror("Failed to read file");
        free(data);
        fclose(file);
        return NULL;
    }
    
    fclose(file);
    return data;
}

// // 将双精度数据写入二进制文件
// static int write_vector_to_file(const char* filename, const double* data, size_t n) {
//     FILE* file = fopen(filename, "wb");
//     if (!file) {
//         perror("Failed to open file for writing");
//         return -1;
//     }
    
//     if (fwrite(data, sizeof(double), n, file) != n) {
//         perror("Failed to write file");
//         fclose(file);
//         return -1;
//     }
    
//     fclose(file);
//     return 0;
// }

static void test(int n, double* alpha, const char* fileX, const char* fileY,const char* fileResult){
    blasHandle_t handle;
    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasCreate(&handle));

    unsigned sizeX, sizeY, sizeResult;
    double* hX = read_vector_from_file(fileX, &sizeX);
    double* hY = read_vector_from_file(fileY, &sizeY);
    double* result = read_vector_from_file(fileResult, &sizeResult);

    if (!hX || !hY || !result) {
        fprintf(stderr, "%s(%u): Failed to read input and output files\n", __FUNCTION__, __LINE__);
        fflush(stderr);
       
        free(hX);
        free(hY);
        free(result);
        return ;
    }
    
    if (!((sizeX == sizeY) && (sizeX == sizeResult) && (sizeX == (unsigned)n))) {
        fprintf(stderr, "%s(%u): x ,y and result vectors must have the same size, and they all should equal n.\n", __FUNCTION__, __LINE__);
        fflush(stderr);
        
        free(hX);
        free(hY);
        free(result);
        return ;
    }
    
    
    // 分配设备内存
    double *dX, *dY, *dResult;

    cudaMalloc(&dX, n * sizeof(double));
    cudaMalloc(&dY, n * sizeof(double));
    cudaMalloc(&dResult, n * sizeof(double));
    

    double* hResult = (double*)malloc(n * sizeof(double));

    // 拷贝数据到设备
    cudaMemcpy(dX, hX, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY, n * sizeof(double), cudaMemcpyHostToDevice);

    launchDaxpy(handle, n, alpha, dX, dY, dResult);

    // 将结果拷贝回主机
    cudaMemcpy(hResult, dResult, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    ASSERT_EQ(true, cmpData(hResult, result, n));

    ASSERT_EQ(BLAS_STATUS_SUCCESS, blasDestroy(handle));
    
    // 释放资源
    free(hX);
    free(hY);
    free(hResult);
    free(result);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dResult);
    

}

TEST(BlasDaxpyKernelTest,SmallScaleTest){
    
    double alpha = 32.678;
    double* dAlpha;
    cudaMalloc(&dAlpha, sizeof(double));
    cudaMemcpy(dAlpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);

    test(1000, dAlpha, "/workspace/gmres/tests/daxpy/data/x_32p678000.bin",
    "/workspace/gmres/tests/daxpy/data/y_32p678000.bin",
    "/workspace/gmres/tests/daxpy/data/result_32p678000.bin");
    
}