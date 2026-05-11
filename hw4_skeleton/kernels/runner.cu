#include <stdexcept>
#include <cublas_v2.h>
using namespace std;

#include "0_cublas.cu"

// // hint 1
// #include "naive_kernel.cu"

// // hint 2
// #include "shared_memory_tiling.cu"

// // hint 3+4
// #include "2d_thread_bloacktiling_register_cahcing.cu"

// hint 5+6+7
#include "student_kernel.cu"

void run_kernel(int kernel_num, int M, int N, int K, float alpha,
                float* A, float* B, float beta, float* C,
                cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            runCublasFP32(M, N, K, alpha, A, B, beta, C, handle);
            break;
        case 1:
            runStudent(M, N, K, alpha, A, B, beta, C);
            break;
        default:
            throw invalid_argument("unknown kernel number\n");
    }
}
