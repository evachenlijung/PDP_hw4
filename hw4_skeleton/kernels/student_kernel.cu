#pragma once
#include "../cuda_utils.h"
#include "../math_utils.h"
#include <stdio.h>

// B: Block
// T: Thread
// M: C row 數
// N: C column 數
// K: A column 數 & B row 數

#define BM 128
#define BN 128
#define BK 8

#define TM 8
#define TN 8

#define BLOCK_THREADS ((BM/TM) * (BN/TN))

// #define NAIVE_BLOCK 32
// #define TILE_SIZE 32

// =============================================================================
// HW4: CUDA Matmul Optimization
//
// Implement your optimized single-precision GEMM here:
//     C = alpha * (A @ B) + beta * C      (row-major, M = N = K per test size)
//
// Hard rules (violation => 0 points for the performance / rank components):
//   1. DO NOT call cuBLAS, cuDNN, CUTLASS, or any vendor GEMM library.
//   2. DO NOT modify any file outside kernels/student_kernel.cu.
//   3. The signature of runStudent() MUST remain unchanged — TA's grading
//      harness calls it directly.
//
// Your kernel is verified against the cuBLAS reference (tolerance 1e-2) for
// all sizes in {128, 256, 512, 1024, 2048, 4096}. Failing any size zeroes out
// the performance and rank components of the grade.
// =============================================================================

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    // TODO: implement your kernel.

    // 算thread負責的C矩陣座標[row][col]
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    // int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // count thread position in the block
    int tx = threadIdx.x % (BN/TN);
    int ty = threadIdx.x / (BM/TM);

    // 這個 block 負責的 C 區塊左上角位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // // 在 shared memory 開空間存 A tile, B tile
    // // __shared__ 表示這塊記憶體是整個 block 共用的
    // __shared__ float sA[TILE_SIZE][TILE_SIZE];
    // __shared__ float sB[TILE_SIZE][TILE_SIZE];

    //  A tile 存轉置
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    // 暫存此 thread 計算結果
    float C_tile[TM][TN] = {0.0f};
    float regA[TM]; // read a column from sA
    float regB[TN]; // read a row from sB

    int a_stride = BLOCK_THREADS / BK;
    int b_stride = BLOCK_THREADS / BN;

    // 搬 A, B 的起始位置
    int a_start_row = threadIdx.x / BK;
    int a_start_col = threadIdx.x % BK;
    int b_start_row = threadIdx.x / BN;
    int b_start_col = threadIdx.x % BN;

    // 沿 K 方向每次處理 BK 寬的寬帶
    for(int t=0; t < CEIL_DIV(K, BK); t++){
        // 搬 A tile 進 shared memory
        for(int i=0; i<BM; i+=a_stride){
            int global_row = block_row + a_start_row + i;
            int global_col = BK * t + a_start_col;
            sA[a_start_col][a_start_row + i] = 
                (global_row < M && global_col < K) 
                ? A[K * global_row + global_col] : 0.0f;
        }

        // 搬 B tile 進 shared memory
        for(int i=0; i<BK; i+=b_stride){
            int global_row = BK * t + b_start_row + i;
            int global_col = block_col + b_start_col;
            sB[b_start_row + i][b_start_col] = 
                (global_row < K && global_col < N) 
                ? B[N * global_row + global_col] : 0.0f;
        }

        __syncthreads();
        
        // compute tile
        for(int k=0; k < BK; k++){
            for(int m = 0; m<TM; m++){
                regA[m] = sA[k][TM * ty + m];
            }
            for(int n = 0; n<TN; n++){
                regB[n] = sB[k][TN * tx + n];
            }
            for(int m = 0; m<TM; m++)
                for(int n = 0; n<TN; n++)
                    C_tile[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // 結果寫回 global memory
    for(int m=0; m<TM; m++){
        for(int n=0; n<TN;  n++){
            int global_row = block_row + TM * ty + m;
            int global_col = block_col + TN * tx + n;
            if(global_row < M && global_col < N){
                C[global_row * N + global_col] = 
                    alpha * C_tile[m][n] + beta * C[global_row * N + global_col];
            }
        }
    } 

    // // 邊界檢查
    // if(row >= M || col >= N) return;

    // // 算內積: A[row] * B col 行
    // float sum = 0.0f;
    // for(int k=0; k<K; k++){
    //     sum += A[K * row + k]*B[N * k + col];
    // }

    // // 沿 K 方向每次處理一個 TILE_SIZE 寬的寬帶
    // float sum = 0.0f;
    // for(int t=0; t < CEIL_DIV(K, TILE_SIZE); t++){
    //     // 搬 A tile, B tile data
    //     int a_col = TILE_SIZE * t + threadIdx.x;
    //     if(a_col < K){
    //         sA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
    //     } else {
    //         sA[threadIdx.y][threadIdx.x] = 0.0f;
    //     }

    //     int b_row = TILE_SIZE * t + threadIdx.y;
    //     if(b_row < K){
    //         sB[threadIdx.y][threadIdx.x] = B[N * b_row + col];
    //     } else {
    //         sB[threadIdx.y][threadIdx.x] = 0.0f;
    //     }

    //     __syncthreads();

    //     for(int k=0; k<TILE_SIZE; k++){
    //         sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    //     }

    //     __syncthreads();

    // }

    // // 寫回
    // C[N * row + col] = alpha * sum + beta * C[N * row + col];
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    // TODO: configure grid/block dimensions and launch StudentKernel.

    // Grid 大小：需要多少個 block 才能覆蓋整個 M×N 矩陣
    // // Block 大小：每個 block 有 32×32 = 1024 個 thread

    // dim3 gridDim(CEIL_DIV(N, NAIVE_BLOCK), CEIL_DIV(M, NAIVE_BLOCK));
    // dim3 blockDim(NAIVE_BLOCK, NAIVE_BLOCK);
    
    // dim3 gridDim(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));
    // dim3 blockDim(TILE_SIZE, TILE_SIZE);

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(BLOCK_THREADS);

    StudentKernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
