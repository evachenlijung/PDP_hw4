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
#define BK 16

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

    // count thread position in the block
    int tx = threadIdx.x % (BN/TN);
    int ty = threadIdx.x / (BM/TM);

    // 這個 block 負責的 C 區塊左上角位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 在 shared memory 開空間存 A tile, B tile
    //  A tile 存轉置
    __shared__ float sA[BK][BM];
    __shared__ float sB[BK][BN];

    // 暫存此 thread 計算結果
    float C_tile[TM][TN] = {0.0f};
    float regA[TM]; // read a column from sA
    float regB[TN]; // read a row from sB

    int a_stride = BLOCK_THREADS / (BK/4);
    int b_stride = BLOCK_THREADS / (BN/4);

    // // 搬 A, B 的起始位置
    // move A, B, 4 floats per step
    int a_load_row = threadIdx.x / (BK/4);
    int a_load_col = threadIdx.x % (BK/4);
    int b_load_row = threadIdx.x / (BN/4);
    int b_load_col = threadIdx.x % (BN/4);

    int a_row_base = block_row + a_load_row;
    // int b_col_base = block_col + b_load_col;

    // 沿 K 方向每次處理 BK 寬的寬帶
    for(int t=0; t < CEIL_DIV(K, BK); t++){
        // // 搬 A tile 進 shared memory
        int a_global_col = BK * t + a_load_col * 4;
        for(int i=0; i<BM; i+=a_stride){
            int a_global_row = a_row_base + i;
            if(a_global_row < M && a_global_col + 3 < K){
                float4 tmp = *reinterpret_cast<float4*>(&A[a_global_row * K + a_global_col]);
                sA[a_load_col * 4 + 0][a_load_row + i] = tmp.x;
                sA[a_load_col * 4 + 1][a_load_row + i] = tmp.y;
                sA[a_load_col * 4 + 2][a_load_row + i] = tmp.z;
                sA[a_load_col * 4 + 3][a_load_row + i] = tmp.w;
            }else{
                sA[a_load_col * 4 + 0][a_load_row + i] = 0.0f;
                sA[a_load_col * 4 + 1][a_load_row + i] = 0.0f;
                sA[a_load_col * 4 + 2][a_load_row + i] = 0.0f;
                sA[a_load_col * 4 + 3][a_load_row + i] = 0.0f;
            }
        }

        // // 搬 B tile 進 shared memory
        int b_row_base = BK * t + b_load_row;
        int b_global_col = block_col + b_load_col * 4;
        for(int i=0; i<BK; i+=b_stride){
            int b_global_row = b_row_base + i;
            if(b_global_row < K && b_global_col + 3 < N){
                *reinterpret_cast<float4*>(&sB[b_load_row + i][b_load_col * 4]) 
                    = *reinterpret_cast<float4*>(&B[b_global_row * N + b_global_col]);
            }else{
                *reinterpret_cast<float4*>(&sB[b_load_row + i][b_load_col * 4]) 
                    = make_float4(0,0,0,0);
            }
        }

        __syncthreads();
        
        // compute tile
        int sa_col_base = TM * ty;
        int sb_col_base = TN * tx;
        for(int k=0; k < BK; k++){
            for(int m = 0; m<TM; m++){
                regA[m] = sA[k][sa_col_base + m];
            }
            for(int n = 0; n<TN; n++){
                regB[n] = sB[k][sb_col_base + n];
            }
            for(int m = 0; m<TM; m++)
                for(int n = 0; n<TN; n++)
                    C_tile[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // // 結果寫回 global memory
    // int c_tile_row_base = block_row + TM * ty;
    // int c_tile_col_base = block_col + TN * tx;
    // for(int m=0; m<TM; m++){
    //     for(int n=0; n<TN;  n++){
    //         int global_row = c_tile_row_base + m;
    //         int global_col = c_tile_col_base + n;
    //         if(global_row < M && global_col < N){
    //             C[global_row * N + global_col] = 
    //                 alpha * C_tile[m][n] + beta * C[global_row * N + global_col];
    //         }
    //     }
    // }

    // 計算該 thread 在 Global Memory 中起始的索引
    int c_tile_row_base = block_row + TM * ty;
    int c_tile_col_base = block_col + TN * tx;

    for (int m = 0; m < TM; m++) {
        int global_row = c_tile_row_base + m;
        int global_col = c_tile_col_base;

        // 檢查 Row 是否在邊界內 (M)
        // 假設 N 始終是 128 的倍數，則 global_col 不會越界
        if (global_row < M) {
            // 定義指向 Global Memory 中 C 的 float4 指標
            // 因為 TN=8，我們需要處理兩個 float4 (共 8 個 float)
            float4* c_ptr = reinterpret_cast<float4*>(&C[global_row * N + global_col]);

            // 1. 讀取舊的 C 值 (Vectorized Load)
            float4 old_c1 = c_ptr[0];
            float4 old_c2 = c_ptr[1];

            // 2. 打包計算結果並套用 alpha / beta
            // 處理前 4 個元素 (n=0~3)
            float4 res1;
            res1.x = alpha * C_tile[m][0] + beta * old_c1.x;
            res1.y = alpha * C_tile[m][1] + beta * old_c1.y;
            res1.z = alpha * C_tile[m][2] + beta * old_c1.z;
            res1.w = alpha * C_tile[m][3] + beta * old_c1.w;

            // 處理後 4 個元素 (n=4~7)
            float4 res2;
            res2.x = alpha * C_tile[m][4] + beta * old_c2.x;
            res2.y = alpha * C_tile[m][5] + beta * old_c2.y;
            res2.z = alpha * C_tile[m][6] + beta * old_c2.z;
            res2.w = alpha * C_tile[m][7] + beta * old_c2.w;

            // 3. 寫回結果 (Vectorized Store)
            c_ptr[0] = res1;
            c_ptr[1] = res2;
        }
    }    

}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    // TODO: configure grid/block dimensions and launch StudentKernel.
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(BLOCK_THREADS);

    StudentKernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
