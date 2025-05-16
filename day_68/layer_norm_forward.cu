#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Layer Normalization Forward Pass Kernel
__global__ void layer_norm_forward_kernel(
    const float* input,    // Input tensor [N, D]
    float* output,         // Output tensor [N, D]
    const float* gamma,    // Scale parameter [D]
    const float* beta,     // Shift parameter [D]
    float* mean,           // Mean (per sample) [N]
    float* variance,       // Variance (per sample) [N]
    int N,                 // Batch size
    int D,                 // Feature dimension
    float eps              // Epsilon for numerical stability
) {
    // Each block handles one sample in the batch
    int n = blockIdx.x;
    
    // Shared memory for parallel reduction
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sq_sum = shared_data + blockDim.x;
    
    // Step 1: Calculate mean and variance using parallel reduction
    float thread_sum = 0;
    float thread_sq_sum = 0;
    
    // Each thread processes multiple elements if needed
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = input[n * D + d];
        thread_sum += val;
        thread_sq_sum += val * val;
    }
    
    // Store in shared memory
    shared_sum[threadIdx.x] = thread_sum;
    shared_sq_sum[threadIdx.x] = thread_sq_sum;
    __syncthreads();
    
    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // First thread finalizes the mean and variance calculation
    if (threadIdx.x == 0) {
        mean[n] = shared_sum[0] / D;
        variance[n] = (shared_sq_sum[0] / D) - (mean[n] * mean[n]);
    }
    __syncthreads();
    
    // Step 2: Normalize and apply scale/shift
    float sample_mean = mean[n];
    float sample_variance = variance[n];
    float inv_std = rsqrtf(sample_variance + eps);
    
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float normalized = (input[n * D + d] - sample_mean) * inv_std;
        output[n * D + d] = gamma[d] * normalized + beta[d];
    }
}

int main() {
    // Sample dimensions for layer norm
    int batch_size = 32;   // N
    int feature_dim = 512; // D
    float eps = 1e-5f;
    
    size_t input_size = batch_size * feature_dim;
    size_t feature_size = feature_dim;
    
    // Allocate host memory
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_output = (float*)malloc(input_size * sizeof(float));
    float *h_gamma = (float*)malloc(feature_size * sizeof(float));
    float *h_beta = (float*)malloc(feature_size * sizeof(float));
    float *h_mean = (float*)malloc(batch_size * sizeof(float));
    float *h_variance = (float*)malloc(batch_size * sizeof(float));
    
    // Initialize input with random values
    for (int i = 0; i < input_size; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Values between -1 and 1
    }
    
    // Initialize gamma and beta
    for (int i = 0; i < feature_dim; i++) {
        h_gamma[i] = 1.0f;  // Default scale = 1
        h_beta[i] = 0.0f;   // Default shift = 0
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta, *d_mean, *d_variance;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    cudaMalloc(&d_gamma, feature_size * sizeof(float));
    cudaMalloc(&d_beta, feature_size * sizeof(float));
    cudaMalloc(&d_mean, batch_size * sizeof(float));
    cudaMalloc(&d_variance, batch_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, feature_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, feature_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel
    int threads = 256;
    int shared_mem_size = 2 * threads * sizeof(float);  // Space for sum and squared sum
    
    printf("Running Layer Normalization forward pass...\n");
    
    cudaEventRecord(start);
    layer_norm_forward_kernel<<<batch_size, threads, shared_mem_size>>>(
        d_input, d_output, d_gamma, d_beta, d_mean, d_variance,
        batch_size, feature_dim, eps
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mean, d_mean, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variance, d_variance, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("Forward pass completed in %.3f ms\n", milliseconds);
    printf("Sample outputs from first batch:\n");
    for (int i = 0; i < 5; i++) {
        printf("Output[0,%d] = %.6f\n", i, h_output[i]);
    }
    printf("Mean[0] = %.6f, Variance[0] = %.6f\n", h_mean[0], h_variance[0]);
    
    // Calculate mean and variance of output (should be ~0 and ~1)
    float out_mean = 0.0f;
    float out_var = 0.0f;
    
    for (int d = 0; d < feature_dim; d++) {
        out_mean += h_output[d];
    }
    out_mean /= feature_dim;
    
    for (int d = 0; d < feature_dim; d++) {
        float diff = h_output[d] - out_mean;
        out_var += diff * diff;
    }
    out_var /= feature_dim;
    
    printf("Output statistics (first sample): Mean = %.6f, Variance = %.6f\n", 
           out_mean, out_var);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_mean);
    cudaFree(d_variance);
    
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);
    free(h_mean);
    free(h_variance);
    
    return 0;
}