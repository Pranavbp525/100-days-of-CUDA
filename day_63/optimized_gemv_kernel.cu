#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Optimized GEMV kernel with thread coarsening and shared memory
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void optimized_gemv_kernel(
    const float* A,
    const float* x,
    float* y,
    int M, int N,
    float alpha, float beta)
{
    // Allocate shared memory for input vector x
    __shared__ float x_shared[BLOCK_SIZE];
    
    // Each thread processes ITEMS_PER_THREAD rows
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = thread_id * ITEMS_PER_THREAD;
    
    // Accumulate results for each assigned row
    float results[ITEMS_PER_THREAD] = {0};
    
    // Process the matrix in tiles along the columns
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Load a tile of x into shared memory
        int col = tile * BLOCK_SIZE + threadIdx.x;
        if (col < N) {
            x_shared[threadIdx.x] = x[col];
        } else {
            x_shared[threadIdx.x] = 0.0f;
        }
        
        __syncthreads();  // Ensure all threads have loaded x
        
        // Compute dot product for each assigned row using the shared memory tile
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            int row = row_start + item;
            if (row < M) {
                for (int i = 0; i < BLOCK_SIZE && (tile * BLOCK_SIZE + i) < N; i++) {
                    results[item] += A[row * N + tile * BLOCK_SIZE + i] * x_shared[i];
                }
            }
        }
        
        __syncthreads();  // Ensure shared memory is not overwritten before all threads finish
    }
    
    // Write results to global memory with alpha and beta scaling
    for (int item = 0; item < ITEMS_PER_THREAD; item++) {
        int row = row_start + item;
        if (row < M) {
            if (beta == 0.0f) {
                y[row] = alpha * results[item];
            } else {
                y[row] = alpha * results[item] + beta * y[row];
            }
        }
    }
}

int main() {
    // Matrix dimensions
    int M = 1024;  // Rows of A and size of y
    int N = 1024;  // Columns of A and size of x
    
    // GEMV parameters
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate host memory
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(M * sizeof(float));
    float *h_y_ref = (float*)malloc(M * sizeof(float));  // For verification
    
    // Initialize matrices with test values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = 1.0f;  // Simple initialization for testing
        }
        h_y[i] = 0.0f;
        h_y_ref[i] = 0.0f;
    }
    
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;  // Simple initialization for testing
    }
    
    // Allocate device memory
    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    // Launch optimized kernel with thread coarsening and shared memory
    const int BLOCK_SIZE = 256;
    const int ITEMS_PER_THREAD = 4;  // Each thread processes 4 rows
    int threads_per_block = BLOCK_SIZE;
    int blocks = (M + ITEMS_PER_THREAD * threads_per_block - 1) / (ITEMS_PER_THREAD * threads_per_block);

    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel
    optimized_gemv_kernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<blocks, threads_per_block>>>(
        d_A, d_x, d_y, M, N, alpha, beta);
    
    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute reference result on CPU for verification
    for (int i = 0; i < M; i++) {
        float dot_product = 0.0f;
        for (int j = 0; j < N; j++) {
            dot_product += h_A[i * N + j] * h_x[j];
        }
        h_y_ref[i] = alpha * dot_product + beta * h_y_ref[i];
    }
    
    // Verify the result
    bool correct = true;
    for (int i = 0; i < M && correct; i++) {
        if (fabs(h_y[i] - h_y_ref[i]) > 1e-5) {
            printf("Error: h_y[%d] = %f, h_y_ref[%d] = %f\n", i, h_y[i], i, h_y_ref[i]);
            correct = false;
        }
    }
    
    // Print results
    printf("Matrix dimensions: A(%d x %d), x(%d), y(%d)\n", M, N, N, M);
    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Calculate performance
    float gflops = (2.0 * M * N) / (milliseconds * 1e6);
    printf("Performance: %.2f GFlop/s\n", gflops);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    // Free host memory
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y_ref);
    
    return 0;
}