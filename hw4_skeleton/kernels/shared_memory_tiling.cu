#pragma once
#include "../cuda_utils.h"
#include "../math_utils.h"

// Tile 大小，兩個都設成 32
// BM = BN = BK = TILE_SIZE
// 每個 block 處理 C 的 TILE_SIZE × TILE_SIZE 區塊
#define TILE_SIZE 32

__global__ void StudentKernel(int M, int N, int K, float alpha,
                               float *A, float *B, float beta, float *C) {

    // 這個 thread 負責的 C 元素座標
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 在 shared memory 開兩塊空間存 A tile 和 B tile
    // __shared__ 表示這塊記憶體是整個 block 共用的
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // 沿著 K 方向，每次處理一個 TILE_SIZE 寬的條帶
    for (int t = 0; t < CEIL_DIV(K, TILE_SIZE); t++) {

        // ── 第一步：把 A tile 和 B tile 搬進 shared memory ──
        // 每個 thread 負責搬一個元素（剛好 TILE_SIZE×TILE_SIZE 個 thread）

        // A tile：第 row 列、第 (t*TILE_SIZE + threadIdx.x) 欄
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;  // 邊界補零

        // B tile：第 (t*TILE_SIZE + threadIdx.y) 列、第 col 欄
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;  // 邊界補零

        // ── 第二步：等所有 thread 都搬完才開始算 ──
        __syncthreads();

        // ── 第三步：用 shared memory 做這個 tile 的內積 ──
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // ── 第四步：算完才能搬下一塊，避免覆蓋還在用的資料 ──
        __syncthreads();
    }

    // 寫回 C
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {

    dim3 gridDim(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
    dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 32×32 = 1024 threads/block

    StudentKernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}