#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Optimized GEMM kernel with thread coarsening (2x2 output per thread)
template <int BLOCK_SIZE>
__global__ void optimized_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Index of the first sub-matrix of A processed by this block
    int aBegin = K * BLOCK_SIZE * by;
    
    // Index of the last sub-matrix of A processed by this block
    int aEnd = aBegin + K - 1;
    
    // Step size used to iterate through sub-matrices of A
    int aStep = BLOCK_SIZE;
    
    // Index of the first sub-matrix of B processed by this block
    int bBegin = BLOCK_SIZE * bx * 2; // * 2 because each thread handles 2 columns
    
    // Step size used to iterate through sub-matrices of B
    int bStep = BLOCK_SIZE * N;
    
    // Each thread computes 2x2 elements of the output matrix
    int cRow = by * BLOCK_SIZE + ty;
    int cCol = bx * BLOCK_SIZE + tx;
    
    // Allocate shared memory for the sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE*2]; // Double width for 2 columns per thread
    
    // Registers for accumulating the result (2x2 block per thread)
    float c[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    // Loop over all sub-matrices of A and B required for this output block
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from global memory to shared memory
        
        // Each thread loads one element of sub-matrix A
        if (ty < BLOCK_SIZE/2) { // Only half the threads needed due to coarsening
            // Each thread loads 2 rows
            int globalRow1 = by * BLOCK_SIZE + ty*2;
            int globalRow2 = globalRow1 + 1;
            int globalCol = a % K + tx;
            
            if (globalRow1 < M && globalCol < K)
                As[ty*2][tx] = A[globalRow1 * K + globalCol];
            else
                As[ty*2][tx] = 0.0f;
                
            if (globalRow2 < M && globalCol < K)
                As[ty*2+1][tx] = A[globalRow2 * K + globalCol];
            else
                As[ty*2+1][tx] = 0.0f;
        }
        
        // Each thread loads one element of sub-matrix B
        if (ty < BLOCK_SIZE) {
            // Each thread loads 2 columns
            int globalRow = b / N + ty;
            int globalCol1 = b % N + tx*2;
            int globalCol2 = globalCol1 + 1;
            
            if (globalRow < K && globalCol1 < N)
                Bs[ty][tx*2] = B[globalRow * N + globalCol1];
            else
                Bs[ty][tx*2] = 0.0f;
                
            if (globalRow < K && globalCol2 < N)
                Bs[ty][tx*2+1] = B[globalRow * N + globalCol2];
            else
                Bs[ty][tx*2+1] = 0.0f;
        }
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Multiply the two matrices together using loop unrolling
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // 2x2 matrix multiplication per thread
            c[0][0] += As[ty*2][k] * Bs[k][tx*2];
            c[0][1] += As[ty*2][k] * Bs[k][tx*2+1];
            c[1][0] += As[ty*2+1][k] * Bs[k][tx*2];
            c[1][1] += As[ty*2+1][k] * Bs[k][tx*2+1];
        }
        
        // Synchronize to avoid reading shared memory before the next iteration
        __syncthreads();
    }
    
    // Write the 2x2 result block to global memory with alpha and beta scaling
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int row = cRow + i;
            int col = cCol + j;
            
            if (row < M && col < N) {
                if (beta == 0.0f)
                    C[row * N + col] = alpha * c[i][j];
                else
                    C[row * N + col] = alpha * c[i][j] + beta * C[row * N + col];
            }
        }
    }
}

int main() {
    // Matrix dimensions
    // M = rows of A and C
    // N = columns of B and C
    // K = columns of A and rows of B
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    // GEMM scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    // Initialize matrices with test values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = 1.0f;  // Simple initialization for testing
        }
    }
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = 1.0f;  // Simple initialization for testing
        }
    }
    
    // Initialize C to zeros
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    const int BLOCK_SIZE = 16;  // Must be a multiple of 2 for our 2x2 thread tiles
    
    // Grid dimensions - half the threads in y direction due to 2x2 tiles
    dim3 dimBlock(BLOCK_SIZE/2, BLOCK_SIZE/2);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Timing using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch kernel
    optimized_gemm_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta);
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Print performance statistics
    double gflops = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("Matrix dimensions: A(%d x %d), B(%d x %d), C(%d x %d)\n", M, K, K, N, M, N);
    printf("Thread block size: %d x %d\n", BLOCK_SIZE/2, BLOCK_SIZE/2);
    printf("Each thread computes a 2x2 output tile\n");
    printf("Performance: %.2f GFlop/s\n", gflops);
    printf("Execution time: %.2f ms\n", milliseconds);
    
    // Verify a corner of the result (for 1.0 initialized matrices, expect K)
    printf("\nC[0,0] = %.1f (expected: %.1f)\n", h_C[0], (float)K);
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}