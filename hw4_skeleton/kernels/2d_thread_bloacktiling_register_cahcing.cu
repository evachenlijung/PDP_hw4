#pragma once
#include "../cuda_utils.h"
#include "../math_utils.h"

// Block tile 大小：一個 block 負責 C 的 BM×BN 區塊
#define BM 128
#define BN 128
#define BK 8
// Thread tile 大小：一個 thread 負責 C 的 TM×TN 區塊
#define TM 8
#define TN 8
// Block 內的 thread 數：(BM/TM) × (BN/TN) = 16×16 = 256
#define BLOCK_THREADS ((BM / TM) * (BN / TN))

__global__ void StudentKernel(int M, int N, int K, float alpha,
                               float *A, float *B, float beta, float *C) {

    // ── 這個 thread 在 block 內的位置 ──
    // 把 1D 的 threadIdx.x 轉成 2D：thread (ty, tx)
    int tx = threadIdx.x % (BN / TN);  // 0..15，對應 C tile 的 col 方向
    int ty = threadIdx.x / (BN / TN);  // 0..15，對應 C tile 的 row 方向

    // ── 這個 block 負責的 C 區塊左上角 ──
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // ── Shared memory ──
    __shared__ float As[BK][BM+1];  // A tile 轉置存，方便後面讀取
    __shared__ float Bs[BK][BN];  // B tile 正常存

    // ── Register：這個 thread 的計算結果暫存 ──
    float C_tile[TM][TN] = {0.0f};  // TM×TN 個累加器，全在 register
    float regA[TM];                  // 從 As 讀出的一個 column
    float regB[TN];                  // 從 Bs 讀出的一個 row

    // ── 每個 thread 負責搬幾個元素進 shared memory ──
    // 總共要搬 BM*BK 個 A 元素和 BK*BN 個 B 元素
    // 由 BLOCK_THREADS(256) 個 thread 平分
    int A_stride = BLOCK_THREADS / BK;   // 每輪 A 搬幾列：256/8 = 32
    int B_stride = BLOCK_THREADS / BN;   // 每輪 B 搬幾列：256/128 = 2

    // 這個 thread 搬 A 時的起始位置
    int a_load_row = threadIdx.x / BK;   // 0..31
    int a_load_col = threadIdx.x % BK;   // 0..7
    // 這個 thread 搬 B 時的起始位置
    int b_load_row = threadIdx.x / BN;   // 0..1
    int b_load_col = threadIdx.x % BN;   // 0..127

    // ── 主迴圈：沿 K 方向每次處理 BK 寬的條帶 ──
    for (int t = 0; t < CEIL_DIV(K, BK); t++) {

        // 搬 A tile 進 shared memory（轉置：As[k][m] = A[m][k]）
        // 轉置的原因：之後讀 As 時是讀同一個 k 的所有 m，連續存才不會 bank conflict
        for (int i = 0; i < BM; i += A_stride) {
            int global_row = block_row + a_load_row + i;
            int global_col = t * BK + a_load_col;
            As[a_load_col][a_load_row + i] =
                (global_row < M && global_col < K)
                ? A[global_row * K + global_col]
                : 0.0f;
        }

        // 搬 B tile 進 shared memory（正常存：Bs[k][n] = B[k][n]）
        for (int i = 0; i < BK; i += B_stride) {
            int global_row = t * BK + b_load_row + i;
            int global_col = block_col + b_load_col;
            Bs[b_load_row + i][b_load_col] =
                (global_row < K && global_col < N)
                ? B[global_row * N + global_col]
                : 0.0f;
        }

        __syncthreads();  // 等所有 thread 搬完

        // ── 用 shared memory 計算這個 tile ──
        for (int k = 0; k < BK; k++) {
            // 把 As 的第 k 列（TM 個值）讀進 register
            for (int m = 0; m < TM; m++)
                regA[m] = As[k][ty * TM + m];
            // 把 Bs 的第 k 列（TN 個值）讀進 register
            for (int n = 0; n < TN; n++)
                regB[n] = Bs[k][tx * TN + n];
            // TM×TN 次 FMA，全在 register，不碰 memory
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    C_tile[m][n] += regA[m] * regB[n];
        }

        __syncthreads();  // 算完再搬下一塊
    }

    // ── 把結果寫回 global memory ──
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int global_row = block_row + ty * TM + m;
            int global_col = block_col + tx * TN + n;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] =
                    alpha * C_tile[m][n] + beta * C[global_row * N + global_col];
            }
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {

    // Grid：需要多少個 block 覆蓋整個 C
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    // Block：BLOCK_THREADS 個 thread，排成 1D
    dim3 blockDim(BLOCK_THREADS);

    StudentKernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}