# PDP HW4 — CUDA SGEMM Optimization: Context Summary

## Assignment Overview

- **Course**: Parallel and Distributed Programming (PDP), NTU, Spring 2026
- **Task**: Implement an optimized single-precision GEMM (SGEMM) CUDA kernel
- **Formula**: `C = alpha * (A @ B) + beta * C` (row-major, M=N=K per test)
- **Test sizes**: {128, 256, 512, 1024, 2048, 4096}
- **Target GPU**: NVIDIA Tesla V100-SXM2-32GB (Compute Capability 7.0, sm_70)
- **Deadline**: 2026-05-22
- **Environment**: TWCC HPC cluster, Slurm scheduler, Project ID `ACD115083`
- **Only file to modify**: `kernels/student_kernel.cu`
- **Forbidden**: cuBLAS, cuDNN, CUTLASS, any vendor GEMM library

## Performance Tiers (at size=4096)

| Tier | Threshold (GFLOPS) | Cumulative Score |
|------|--------------------|-----------------|
| T1   | ≥ 200              | +20%            |
| T2   | ≥ 2050             | +25%            |
| T3   | ≥ 3700             | +30%            |
| T4   | ≥ 7600             | +37%            |
| T5   | ≥ 9500             | +40%            |

## Build & Run Commands

```bash
# On TWCC login node (ln01.twcc.ai)
module load cuda
make clean && make

# Run via Slurm
srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 5 ./main
```

## Optimization Progress

| Step | Technique | 4096 GFLOPS | Notes |
|------|-----------|-------------|-------|
| 0 | Naïve kernel (1 thread = 1 C element) | 2283 | Already passes T2 |
| 1 | Shared memory tiling (TILE_SIZE=32) | 4162 | Passes T3 |
| 2 | 2D thread blocktiling + register caching | **11017** | Passes T5 |
| 3 | float4 vectorized loads + BK=16 | in progress | Targeting higher rank |

## Current Kernel Design (`kernels/student_kernel.cu`)

### Parameters

```cuda
#define BM 128           // Block tile rows (C rows per block)
#define BN 128           // Block tile cols (C cols per block)
#define BK 16            // Block tile depth (K-direction strip width)
#define TM 8             // Thread tile rows (C rows per thread)
#define TN 8             // Thread tile cols (C cols per thread)
#define BLOCK_THREADS ((BM/TM) * (BN/TN))  // = 256 threads per block
```

### Shared Memory Layout

```cuda
__shared__ float sA[BK][BM];   // A tile stored TRANSPOSED: sA[k][m]
__shared__ float sB[BK][BN];   // B tile stored normally:   sB[k][n]
```

`sA` is transposed to avoid bank conflicts when reading columns of A.

### Thread Index Convention (1D block)

```cuda
int tx = threadIdx.x % (BN/TN);   // 0..15, column direction of C tile
int ty = threadIdx.x / (BN/TN);   // 0..15, row direction of C tile
// Each thread computes C[ty*TM .. ty*TM+7][tx*TN .. tx*TN+7]
```

**Important**: block is 1D (`dim3(BLOCK_THREADS)`), NOT 2D. All indexing uses `threadIdx.x`.

### Data Loading Indices (float4 version, BK=16)

```cuda
int a_stride   = BLOCK_THREADS / BK;           // 256/16 = 16
int b_stride   = BLOCK_THREADS / BN;           // 256/128 = 2

int a_load_row = threadIdx.x / (BK/4) % a_stride;  // 0..15 (BM direction)
int a_load_col = threadIdx.x % (BK/4);              // 0..3  (4 float4s cover BK=16)
int b_load_row = threadIdx.x / (BN/4) % b_stride;  // 0..1  (BK direction)
int b_load_col = threadIdx.x % (BN/4);              // 0..31 (32 float4s cover BN=128)
```

### Main Loop Structure

```
for t in 0..K/BK:
    1. Load A tile into sA (transposed) using float4
       - Each thread loads 4 floats per iteration, BM/a_stride iterations
       - sA[a_load_col*4 + 0..3][a_load_row + i] = float4 from A
    2. Load B tile into sB using float4
       - Each thread loads 4 floats per iteration, BK/b_stride iterations
       - sB[b_load_row + i][b_load_col*4 .. +3] = float4 from B
    3. __syncthreads()
    4. Compute: for k in 0..BK:
         regA[0..TM-1] = sA[k][ty*TM .. ty*TM+TM-1]
         regB[0..TN-1] = sB[k][tx*TN .. tx*TN+TN-1]
         C_tile[m][n] += regA[m] * regB[n]   (TM×TN FMAs)
    5. __syncthreads()
6. Write C_tile back to global memory with alpha/beta scaling
```

### Launch Configuration

```cuda
dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
dim3 blockDim(BLOCK_THREADS);   // 256 threads, 1D
StudentKernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

## Key Concepts Covered

### CUDA Fundamentals
- `cudaMalloc` / `cudaFree` / `cudaMemcpy` (HostToDevice, DeviceToHost, DeviceToDevice)
- `cudaEvent_t` for timing: `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`
- `cudaDeviceSynchronize()` — blocks CPU until all GPU work completes
- Kernel launch syntax: `kernel<<<gridDim, blockDim, sharedMem, stream>>>(args)`
- `__shared__` memory, `__syncthreads()`
- 1D vs 2D block layout (hardware is always 1D underneath)

### Optimization Techniques Applied
1. **Global memory coalescing**: threads in a warp access contiguous addresses
2. **Shared memory tiling**: stage A/B tiles into shared memory, reuse across threads
3. **1D thread blocktiling**: each thread computes TM output elements, increases arithmetic intensity
4. **2D thread blocktiling + register caching**: each thread computes TM×TN micro-tile in registers; arithmetic intensity = TM×TN/(TM+TN) = 4× for TM=TN=8
5. **Bank-conflict-free**: transposing sA (storing as `sA[BK][BM]`) avoids column-access bank conflicts; no padding needed since BM=128 is a multiple of 32
6. **Vectorized loads (float4)**: one 128-bit instruction loads 4 floats; `*reinterpret_cast<float4*>(&A[idx])` — requires 16-byte alignment (index must be multiple of 4)

### FMA (Fused Multiply-Add)
- One hardware instruction: `a = a + b * c`
- GFLOPS formula: `2 * M * N * K / time / 1e9` — the `2` counts 1 multiply + 1 add per FMA

## Common Bugs Encountered & Fixed

| Bug | Symptom | Cause | Fix |
|-----|---------|-------|-----|
| `#prgama once` typo | Compile error | Typo in pragma | `#pragma once` |
| `#define BLOCK_THRAEDS` typo | `BLOCK_THREADS` undefined | Typo in define | `BLOCK_THREADS` |
| `ty = threadIdx.y % ...` | Verification failed (diff ~376) | Block is 1D, `threadIdx.y` always 0 | `ty = threadIdx.x / (BN/TN)` |
| `ty = threadIdx.x %` (same as tx) | Verification failed (diff ~72) | Both tx and ty used `%` | `ty` uses `/`, `tx` uses `%` |
| `b_load_row + i` out of bounds | Illegal memory access | After BK 8→16, `b_load_row` max=7, `i` max=14 → index 21 > BK=16 | Add `% b_stride` to `b_load_row` and `a_load_row` |

## SSH / Cluster Setup

- **Login node**: `r14922112@ln01.twcc.ai`
- **SSH config** (`C:\Users\eva\.ssh\config` on Windows):
  ```
  Host twcc
      HostName ln01.twcc.ai
      User r14922112
      IdentityFile C:\Users\eva\.ssh\id_ed25519
      ServerAliveInterval 60
      MACs hmac-sha2-256
  ```
- **VSCode setting** to fix WSL/OpenSSH conflict:
  `"remote.SSH.path": "C:\\WINDOWS\\System32\\OpenSSH\\ssh.exe"`
- **SCP from WSL**:
  ```bash
  scp -o MACs=hmac-sha2-256 kernels/student_kernel.cu r14922112@ln01.twcc.ai:~/hw4_skeleton/kernels/
  ```

## Current Status

- ✅ Verification passing for all 6 sizes
- ✅ T5 (≥9500 GFLOPS) achieved at step 2 (11017 GFLOPS @ size=4096)
- 🔄 Applying float4 vectorized loads + BK=16 for rank competition
- 📝 Report to be written after optimization is complete
