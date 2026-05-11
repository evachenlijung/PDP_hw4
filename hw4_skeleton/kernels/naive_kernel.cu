#include "../math_utils.h"
#include <stdio.h>

// ──────────────────────────────────────────────
// Naïve kernel：每個 thread 算一個 C[row][col]
// ──────────────────────────────────────────────

#define NAIVE_BLOCK 32   // 每個 block 是 32×32 個 thread

__global__ void StudentKernel(int M, int N, int K, float alpha,
                               float *A, float *B, float beta, float *C) {

    // 1. 算出這個 thread 負責的 C 矩陣座標
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 邊界檢查：M、N 不一定是 block size 的整數倍
    if (row >= M || col >= N) return;

    // 3. 計算內積：A 的第 row 列 × B 的第 col 行
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // 矩陣是 row-major：A[row][k] = A[row * K + k]
        sum += A[row * K + k] * B[k * N + col];
    }

    // 4. 套上 alpha/beta，寫回 C
    //    公式：C = alpha * (A @ B) + beta * C
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {

    // Grid 大小：需要多少個 block 才能覆蓋整個 M×N 矩陣
    // CEIL_DIV(M, 32) = (M + 31) / 32，避免除不盡時少算
    dim3 gridDim(CEIL_DIV(N, NAIVE_BLOCK), CEIL_DIV(M, NAIVE_BLOCK));

    // Block 大小：每個 block 有 32×32 = 1024 個 thread
    dim3 blockDim(NAIVE_BLOCK, NAIVE_BLOCK);

    StudentKernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
