#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GEMM kernel with shared memory tiling
template <int BLOCK_SIZE>
__global__ void gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Shared memory for the sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of the output sub-matrix
    int row = blockRow * BLOCK_SIZE + threadIdx.y;
    int col = blockCol * BLOCK_SIZE + threadIdx.x;
    
    // Accumulate result for C[row][col]
    float sum = 0.0f;
    
    // Loop over all sub-matrices of A and B
    for (int k = 0; k < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        
        // Load sub-matrices from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        if (row < M && k * BLOCK_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (k * BLOCK_SIZE + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Multiply the two sub-matrices together
        for (int e = 0; e < BLOCK_SIZE; e++) {
            sum += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        }
        
        // Synchronize to ensure next iteration doesn't overwrite shared memory
        __syncthreads();
    }
    
    // Write the result to C
    if (row < M && col < N) {
        if (beta == 0) {
            C[row * N + col] = alpha * sum;
        } else {
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

int main() {
    // Matrix dimensions
    int M = 1024;  // Rows of A and C
    int N = 1024;  // Columns of B and C
    int K = 1024;  // Columns of A and rows of B
    
    // GEMM parameters
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = 1.0f; // Simple initialization for testing
        }
    }
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = 1.0f; // Simple initialization for testing
        }
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_C[i * N + j] = 0.0f;
        }
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    const int BLOCK_SIZE = 32;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel
    gemm_kernel<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    
    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing and performance information
    printf("Matrix sizes: A(%d,%d), B(%d,%d), C(%d,%d)\n", M, K, K, N, M, N);
    printf("Block size: %d\n", BLOCK_SIZE);
    printf("Performance: %.2f GFlop/s\n", 
           (2.0 * M * N * K + 3.0 * M * N) / (milliseconds * 1e6));
    printf("Execution time: %.3f ms\n", milliseconds);
    
    // Verify result (for small matrices)
    if (M <= 10 && N <= 10 && K <= 10) {
        printf("\nMatrix C:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.1f ", h_C[i * N + j]);
            }
            printf("\n");
        }
    } else {
        // For large matrices, just check a sample element
        printf("\nC[0,0] = %.1f\n", h_C[0]);
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}