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

// Layer Normalization Backward Pass - Compute gamma & beta gradients
__global__ void layer_norm_backward_params_kernel(
    const float* grad_output,  // Upstream gradient [N, D]
    const float* input,        // Input from forward pass [N, D]
    const float* mean,         // Mean from forward pass [N]
    const float* variance,     // Variance from forward pass [N]
    float* grad_gamma,         // Gradient w.r.t. gamma [D]
    float* grad_beta,          // Gradient w.r.t. beta [D]
    int N,                     // Batch size
    int D,                     // Feature dimension
    float eps                  // Epsilon for numerical stability
) {
    // Each block handles one feature dimension
    int d = blockIdx.x;
    
    // Shared memory for parallel reduction
    extern __shared__ float shared_data[];
    float* shared_dgamma = shared_data;
    float* shared_dbeta = shared_data + blockDim.x;
    
    // Each thread's partial gradient accumulation
    float thread_dgamma = 0.0f;
    float thread_dbeta = 0.0f;
    
    // Loop over batch
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float inv_std = rsqrtf(variance[n] + eps);
        float normalized = (input[n * D + d] - mean[n]) * inv_std;
        
        // Gradients w.r.t. gamma and beta
        thread_dgamma += grad_output[n * D + d] * normalized;
        thread_dbeta += grad_output[n * D + d];
    }
    
    // Store in shared memory
    shared_dgamma[threadIdx.x] = thread_dgamma;
    shared_dbeta[threadIdx.x] = thread_dbeta;
    __syncthreads();
    
    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_dgamma[threadIdx.x] += shared_dgamma[threadIdx.x + stride];
            shared_dbeta[threadIdx.x] += shared_dbeta[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // First thread writes the gradients
    if (threadIdx.x == 0) {
        grad_gamma[d] = shared_dgamma[0];
        grad_beta[d] = shared_dbeta[0];
    }
}

// Layer Normalization Backward Pass - Compute input gradients
__global__ void layer_norm_backward_input_kernel(
    const float* grad_output,  // Upstream gradient [N, D]
    const float* input,        // Input from forward pass [N, D]
    const float* gamma,        // Scale parameter [D]
    const float* mean,         // Mean from forward pass [N]
    const float* variance,     // Variance from forward pass [N]
    float* grad_input,         // Gradient w.r.t. input [N, D]
    int N,                     // Batch size
    int D,                     // Feature dimension
    float eps                  // Epsilon for numerical stability
) {
    // Each block handles one sample in the batch
    int n = blockIdx.x;
    
    // Shared memory for reductions
    extern __shared__ float shared_data[];
    float* shared_sum_dy = shared_data;         // sum(dy)
    float* shared_sum_dy_xmu = shared_data + blockDim.x;  // sum(dy * (x-μ))
    
    // Calculate average gradient and dot product for this sample
    float thread_sum_dy = 0.0f;
    float thread_sum_dy_xmu = 0.0f;
    
    // Each thread processes multiple elements if needed
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float dy = grad_output[n * D + d] * gamma[d];
        float xmu = input[n * D + d] - mean[n];
        
        thread_sum_dy += dy;
        thread_sum_dy_xmu += dy * xmu;
    }
    
    // Store in shared memory
    shared_sum_dy[threadIdx.x] = thread_sum_dy;
    shared_sum_dy_xmu[threadIdx.x] = thread_sum_dy_xmu;
    __syncthreads();
    
    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum_dy[threadIdx.x] += shared_sum_dy[threadIdx.x + stride];
            shared_sum_dy_xmu[threadIdx.x] += shared_sum_dy_xmu[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Get reduction results
    float sum_dy = shared_sum_dy[0];
    float sum_dy_xmu = shared_sum_dy_xmu[0];
    float inv_std = rsqrtf(variance[n] + eps);
    float inv_var = inv_std * inv_std;
    
    // Calculate gradients for each element
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float dy = grad_output[n * D + d] * gamma[d];
        float xmu = input[n * D + d] - mean[n];
        
        // Full gradient formula:
        // dx = (1. / sqrt(var + eps)) * (dy - mean(dy) - (x-mean) * mean(dy * (x-mean)) / var)
        grad_input[n * D + d] = inv_std * (
            dy - (sum_dy / D) - (xmu * inv_var * sum_dy_xmu / D)
        );
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
    
    float *h_grad_output = (float*)malloc(input_size * sizeof(float));
    float *h_grad_input = (float*)malloc(input_size * sizeof(float));
    float *h_grad_gamma = (float*)malloc(feature_size * sizeof(float));
    float *h_grad_beta = (float*)malloc(feature_size * sizeof(float));
    
    // Initialize input with random values
    for (int i = 0; i < input_size; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Values between -1 and 1
        h_grad_output[i] = ((float)rand() / RAND_MAX) * 0.1f;   // Small random gradients
    }
    
    // Initialize gamma and beta
    for (int i = 0; i < feature_dim; i++) {
        h_gamma[i] = 1.0f;  // Default scale = 1
        h_beta[i] = 0.0f;   // Default shift = 0
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta, *d_mean, *d_variance;
    float *d_grad_output, *d_grad_input, *d_grad_gamma, *d_grad_beta;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    cudaMalloc(&d_gamma, feature_size * sizeof(float));
    cudaMalloc(&d_beta, feature_size * sizeof(float));
    cudaMalloc(&d_mean, batch_size * sizeof(float));
    cudaMalloc(&d_variance, batch_size * sizeof(float));
    
    cudaMalloc(&d_grad_output, input_size * sizeof(float));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));
    cudaMalloc(&d_grad_gamma, feature_size * sizeof(float));
    cudaMalloc(&d_grad_beta, feature_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, feature_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, feature_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // First run forward pass to get required values
    printf("Running Layer Normalization forward pass...\n");
    
    int threads = 256;
    int shared_mem_size = 2 * threads * sizeof(float);
    
    cudaEventRecord(start);
    layer_norm_forward_kernel<<<batch_size, threads, shared_mem_size>>>(
        d_input, d_output, d_gamma, d_beta, d_mean, d_variance,
        batch_size, feature_dim, eps
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Forward pass completed in %.3f ms\n", milliseconds);
    
    // Run backward pass
    printf("Running Layer Normalization backward pass...\n");
    cudaEventRecord(start);
    
    // Step 1: Compute gradients for gamma and beta
    layer_norm_backward_params_kernel<<<feature_dim, threads, shared_mem_size>>>(
        d_grad_output, d_input, d_mean, d_variance,
        d_grad_gamma, d_grad_beta,
        batch_size, feature_dim, eps
    );
    
    // Step 2: Compute gradients for input
    layer_norm_backward_input_kernel<<<batch_size, threads, shared_mem_size>>>(
        d_grad_output, d_input, d_gamma, d_mean, d_variance,
        d_grad_input, batch_size, feature_dim, eps
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Backward pass completed in %.3f ms\n", milliseconds);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_input, d_grad_input, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_gamma, d_grad_gamma, feature_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_beta, d_grad_beta, feature_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print some results for verification
    printf("\nSample gradients:\n");
    printf("grad_input[0,0]: %.6f\n", h_grad_input[0]);
    printf("grad_gamma[0]: %.6f\n", h_grad_gamma[0]);
    printf("grad_beta[0]: %.6f\n", h_grad_beta[0]);
    
    // Verify gradient shapes
    float grad_input_norm = 0.0f;
    float grad_gamma_norm = 0.0f;
    float grad_beta_norm = 0.0f;
    
    for (int i = 0; i < 10; i++) {
        grad_input_norm += h_grad_input[i] * h_grad_input[i];
        grad_gamma_norm += h_grad_gamma[i] * h_grad_gamma[i];
        grad_beta_norm += h_grad_beta[i] * h_grad_beta[i];
    }
    
    printf("\nGradient norms (first 10 elements):\n");
    printf("grad_input_norm: %.6f\n", sqrtf(grad_input_norm));
    printf("grad_gamma_norm: %.6f\n", sqrtf(grad_gamma_norm));
    printf("grad_beta_norm: %.6f\n", sqrtf(grad_beta_norm));
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_gamma);
    cudaFree(d_grad_beta);
    
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);
    free(h_mean);
    free(h_variance);
    free(h_grad_output);
    free(h_grad_input);
    free(h_grad_gamma);
    free(h_grad_beta);
    
    return 0;
}