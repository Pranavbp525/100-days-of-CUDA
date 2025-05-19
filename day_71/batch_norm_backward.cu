#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void batch_norm_forward_kernel(
    const float* input, float* output, const float* gamma, const float* beta,
    float* batch_mean, float* batch_var, float* running_mean, float* running_var,
    int N, int C, int spatial_size, float momentum, float epsilon, bool is_training);

// Batch Normalization Backward Pass Kernel
__global__ void batch_norm_backward_kernel(
    const float* input,           // Input from forward pass [N, C, H, W] or [N, D]
    const float* grad_output,     // Gradient from upstream [N, C, H, W] or [N, D]
    float* grad_input,            // Gradient w.r.t input [N, C, H, W] or [N, D]
    float* grad_gamma,            // Gradient w.r.t gamma [C] or [D]
    float* grad_beta,             // Gradient w.r.t beta [C] or [D]
    const float* gamma,           // Scale parameter [C] or [D]
    const float* batch_mean,      // Mean (from forward pass) [C] or [D]
    const float* batch_var,       // Variance (from forward pass) [C] or [D]
    int N,                        // Batch size
    int C,                        // Channels/Features
    int spatial_size,             // H*W or 1 for fully connected
    float epsilon                 // Epsilon for numerical stability
) {
    // Each block handles one feature/channel
    int c = blockIdx.x;
    
    if (c >= C) return;
    
    // Shared memory for parallel reductions
    extern __shared__ float shared_data[];
    float* shared_dgamma = shared_data;                        // For sum(dy * normalized_x)
    float* shared_dbeta = shared_data + blockDim.x;            // For sum(dy)
    float* shared_sum_dy = shared_data + 2 * blockDim.x;       // For sum(dy)
    float* shared_sum_dy_xmu = shared_data + 3 * blockDim.x;   // For sum(dy * (x-μ))
    
    // Elements per feature
    int elements_per_feature = N * spatial_size;
    
    // Load feature-specific values
    float mean = batch_mean[c];
    float var = batch_var[c];
    float gamma_val = gamma[c];
    float inv_std = rsqrtf(var + epsilon);
    
    // Step 1: Compute gradients for gamma and beta (parameter gradients)
    float thread_dgamma = 0.0f;
    float thread_dbeta = 0.0f;
    
    for (int i = threadIdx.x; i < elements_per_feature; i += blockDim.x) {
        // Calculate global index
        int n = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int idx = ((n * C) + c) * spatial_size + spatial_idx;
        
        // Compute normalized input (can be recomputed or stored in forward pass)
        float normalized = (input[idx] - mean) * inv_std;
        
        // Accumulate gradients for gamma and beta
        float dy = grad_output[idx];
        thread_dgamma += dy * normalized;
        thread_dbeta += dy;
    }
    
    // Store in shared memory
    shared_dgamma[threadIdx.x] = thread_dgamma;
    shared_dbeta[threadIdx.x] = thread_dbeta;
    __syncthreads();
    
    // Reduce within the block for parameter gradients
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_dgamma[threadIdx.x] += shared_dgamma[threadIdx.x + stride];
            shared_dbeta[threadIdx.x] += shared_dbeta[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes parameter gradients
    if (threadIdx.x == 0) {
        grad_gamma[c] = shared_dgamma[0];
        grad_beta[c] = shared_dbeta[0];
    }
    
    // Step 2: Compute intermediate terms for input gradient
    float thread_sum_dy = 0.0f;
    float thread_sum_dy_xmu = 0.0f;
    
    for (int i = threadIdx.x; i < elements_per_feature; i += blockDim.x) {
        // Calculate global index
        int n = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int idx = ((n * C) + c) * spatial_size + spatial_idx;
        
        float dy = grad_output[idx];
        float xmu = input[idx] - mean;
        
        thread_sum_dy += dy;
        thread_sum_dy_xmu += dy * xmu;
    }
    
    // Store in shared memory
    shared_sum_dy[threadIdx.x] = thread_sum_dy;
    shared_sum_dy_xmu[threadIdx.x] = thread_sum_dy_xmu;
    __syncthreads();
    
    // Reduce within the block for intermediate terms
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
    
    // Step 3: Compute gradient for input
    for (int i = threadIdx.x; i < elements_per_feature; i += blockDim.x) {
        // Calculate global index
        int n = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int idx = ((n * C) + c) * spatial_size + spatial_idx;
        
        float dy = grad_output[idx];
        float xmu = input[idx] - mean;
        float norm_factor = 1.0f / float(elements_per_feature);
        
        // Full gradient formula:
        // dx = (gamma/std) * (dy - mean(dy) - (x-mean)/var * mean(dy * (x-mean)))
        grad_input[idx] = gamma_val * inv_std * (
            dy - (sum_dy * norm_factor) - 
            (xmu * inv_std * inv_std * sum_dy_xmu * norm_factor)
        );
    }
}

int main() {
    // Test parameters (same as forward pass)
    int batch_size = 32;        // N
    int channels = 64;          // C
    int height = 28;            // H
    int width = 28;             // W
    int spatial_size = height * width;  // H*W
    float momentum = 0.9f;      // Momentum for running stats
    float epsilon = 1e-5f;      // Epsilon for numerical stability
    
    // Total sizes
    size_t input_size = batch_size * channels * spatial_size;
    size_t param_size = channels;
    
    // Allocate host memory for forward pass
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_output = (float*)malloc(input_size * sizeof(float));
    float *h_gamma = (float*)malloc(param_size * sizeof(float));
    float *h_beta = (float*)malloc(param_size * sizeof(float));
    float *h_batch_mean = (float*)malloc(param_size * sizeof(float));
    float *h_batch_var = (float*)malloc(param_size * sizeof(float));
    float *h_running_mean = (float*)malloc(param_size * sizeof(float));
    float *h_running_var = (float*)malloc(param_size * sizeof(float));
    
    // Additional allocations for backward pass
    float *h_grad_output = (float*)malloc(input_size * sizeof(float));
    float *h_grad_input = (float*)malloc(input_size * sizeof(float));
    float *h_grad_gamma = (float*)malloc(param_size * sizeof(float));
    float *h_grad_beta = (float*)malloc(param_size * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < input_size; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Values between -1 and 1
        h_grad_output[i] = ((float)rand() / RAND_MAX) * 0.1f;   // Small random gradients
    }
    
    for (int i = 0; i < param_size; i++) {
        h_gamma[i] = 1.0f;  // Initialize scale to 1
        h_beta[i] = 0.0f;   // Initialize shift to 0
        h_running_mean[i] = 0.0f;  // Initialize running mean to 0
        h_running_var[i] = 1.0f;   // Initialize running variance to 1
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    float *d_batch_mean, *d_batch_var, *d_running_mean, *d_running_var;
    float *d_grad_output, *d_grad_input, *d_grad_gamma, *d_grad_beta;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    cudaMalloc(&d_gamma, param_size * sizeof(float));
    cudaMalloc(&d_beta, param_size * sizeof(float));
    cudaMalloc(&d_batch_mean, param_size * sizeof(float));
    cudaMalloc(&d_batch_var, param_size * sizeof(float));
    cudaMalloc(&d_running_mean, param_size * sizeof(float));
    cudaMalloc(&d_running_var, param_size * sizeof(float));
    cudaMalloc(&d_grad_output, input_size * sizeof(float));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));
    cudaMalloc(&d_grad_gamma, param_size * sizeof(float));
    cudaMalloc(&d_grad_beta, param_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_mean, h_running_mean, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_var, h_running_var, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // First run forward pass to get statistics needed for backward pass
    int threads = 256;
    int blocks = channels;
    int fwd_shared_mem_size = 2 * threads * sizeof(float);  // For sum and squared sum
    bool is_training = true;
    
    printf("Running Batch Normalization forward pass...\n");
    
    cudaEventRecord(start);
    batch_norm_forward_kernel<<<blocks, threads, fwd_shared_mem_size>>>(
        d_input, d_output, d_gamma, d_beta,
        d_batch_mean, d_batch_var, d_running_mean, d_running_var,
        batch_size, channels, spatial_size, 
        momentum, epsilon, is_training
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float forward_time = 0;
    cudaEventElapsedTime(&forward_time, start, stop);
    printf("Forward pass completed in %.3f ms\n", forward_time);
    
    // Now run backward pass
    int bwd_shared_mem_size = 4 * threads * sizeof(float);  // For dgamma, dbeta, sum_dy, sum_dy_xmu
    
    printf("Running Batch Normalization backward pass...\n");
    
    cudaEventRecord(start);
    batch_norm_backward_kernel<<<blocks, threads, bwd_shared_mem_size>>>(
        d_input, d_grad_output, d_grad_input, d_grad_gamma, d_grad_beta,
        d_gamma, d_batch_mean, d_batch_var,
        batch_size, channels, spatial_size, epsilon
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float backward_time = 0;
    cudaEventElapsedTime(&backward_time, start, stop);
    printf("Backward pass completed in %.3f ms\n", backward_time);
    
    // Copy results back to host
    cudaMemcpy(h_batch_mean, d_batch_mean, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_batch_var, d_batch_var, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_input, d_grad_input, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_gamma, d_grad_gamma, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_beta, d_grad_beta, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print some results for verification
    printf("\nGradient Statistics for First Few Channels:\n");
    for (int c = 0; c < 3 && c < channels; c++) {
        printf("Channel %d: grad_gamma=%.6f, grad_beta=%.6f\n", 
               c, h_grad_gamma[c], h_grad_beta[c]);
    }
    
    // Compute statistics of grad_input for verification
    float grad_input_sum = 0.0f;
    float grad_input_sq_sum = 0.0f;
    
    for (int i = 0; i < 1000 && i < input_size; i++) {
        grad_input_sum += h_grad_input[i];
        grad_input_sq_sum += h_grad_input[i] * h_grad_input[i];
    }
    
    float grad_input_mean = grad_input_sum / 1000.0f;
    float grad_input_var = (grad_input_sq_sum / 1000.0f) - (grad_input_mean * grad_input_mean);
    
    printf("\nGradient Input Statistics (first 1000 elements):\n");
    printf("Mean: %.6f, Variance: %.6f\n", grad_input_mean, grad_input_var);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_batch_mean);
    cudaFree(d_batch_var);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_gamma);
    cudaFree(d_grad_beta);
    
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);
    free(h_batch_mean);
    free(h_batch_var);
    free(h_running_mean);
    free(h_running_var);
    free(h_grad_output);
    free(h_grad_input);
    free(h_grad_gamma);
    free(h_grad_beta);
    
    return 0;
}

__global__ void batch_norm_forward_kernel(
    const float* input,        // Input tensor [N, C, H, W] or [N, D]
    float* output,             // Output tensor [N, C, H, W] or [N, D]
    const float* gamma,        // Scale parameter [C] or [D]
    const float* beta,         // Shift parameter [C] or [D]
    float* batch_mean,         // Mean per feature [C] or [D]
    float* batch_var,          // Variance per feature [C] or [D]
    float* running_mean,       // Running mean (for inference) [C] or [D]
    float* running_var,        // Running variance (for inference) [C] or [D]
    int N,                     // Batch size
    int C,                     // Channels/Features
    int spatial_size,          // H*W or 1 for fully connected
    float momentum,            // Momentum for running stats
    float epsilon,             // Epsilon for numerical stability
    bool is_training           // Whether in training or inference mode
) {
    // Each block handles one feature/channel
    int c = blockIdx.x;
    
    if (c >= C) return;
    
    // Shared memory for parallel reduction
    extern __shared__ float shared_data[];
    float* shared_sum = shared_data;
    float* shared_sq_sum = shared_data + blockDim.x;
    
    // Elements per feature
    int elements_per_feature = N * spatial_size;
    
    // Step 1: Compute mean and variance for this feature/channel
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    if (is_training) {
        // In training mode, compute mean and variance from the current batch
        for (int i = threadIdx.x; i < elements_per_feature; i += blockDim.x) {
            // Calculate global index
            int n = i / spatial_size;
            int spatial_idx = i % spatial_size;
            int idx = ((n * C) + c) * spatial_size + spatial_idx;
            
            float val = input[idx];
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
        
        // First thread computes mean and variance
        if (threadIdx.x == 0) {
            float mean = shared_sum[0] / elements_per_feature;
            float variance = (shared_sq_sum[0] / elements_per_feature) - (mean * mean);
            
            // Store batch statistics
            batch_mean[c] = mean;
            batch_var[c] = variance;
            
            // Update running statistics
            running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
            running_var[c] = momentum * running_var[c] + (1.0f - momentum) * variance;
        }
    }
    
    __syncthreads();
    
    // Step 2: Normalize and apply scale/shift
    float mean, variance;
    
    if (is_training) {
        // Use batch statistics in training mode
        mean = batch_mean[c];
        variance = batch_var[c];
    } else {
        // Use running statistics in inference mode
        mean = running_mean[c];
        variance = running_var[c];
    }
    
    float inv_std = rsqrtf(variance + epsilon);
    float gamma_val = gamma[c];
    float beta_val = beta[c];
    
    // Apply normalization
    for (int i = threadIdx.x; i < elements_per_feature; i += blockDim.x) {
        // Calculate global index
        int n = i / spatial_size;
        int spatial_idx = i % spatial_size;
        int idx = ((n * C) + c) * spatial_size + spatial_idx;
        
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = gamma_val * normalized + beta_val;
    }
}