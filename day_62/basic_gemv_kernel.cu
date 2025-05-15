#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Basic GEMV kernel - one thread per output element
__global__ void basic_gemv_kernel(
    const float* A,       // Input matrix (M x N)
    const float* x,       // Input vector (N)
    float* y,             // Output vector (M)
    int M, int N,         // Matrix dimensions
    float alpha, float beta)  // Scaling factors
{
    // Calculate global thread ID
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within bounds
    if (row < M) {
        float dot_product = 0.0f;
        
        // Compute dot product of row A[row,:] and vector x
        for (int col = 0; col < N; col++) {
            dot_product += A[row * N + col] * x[col];
        }
        
        // Apply scaling: y = alpha * A * x + beta * y
        y[row] = alpha * dot_product + beta * y[row];
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
    int threads_per_block = 256;
    int blocks = (M + threads_per_block - 1) / threads_per_block;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel
    basic_gemv_kernel<<<blocks, threads_per_block>>>(d_A, d_x, d_y, M, N, alpha, beta);
    
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