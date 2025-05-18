#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Batch Normalization Forward Pass Kernel
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

int main() {
    // Test parameters
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
    
    // Allocate host memory
    float *h_input = (float*)malloc(input_size * sizeof(float));
    float *h_output = (float*)malloc(input_size * sizeof(float));
    float *h_gamma = (float*)malloc(param_size * sizeof(float));
    float *h_beta = (float*)malloc(param_size * sizeof(float));
    float *h_batch_mean = (float*)malloc(param_size * sizeof(float));
    float *h_batch_var = (float*)malloc(param_size * sizeof(float));
    float *h_running_mean = (float*)malloc(param_size * sizeof(float));
    float *h_running_var = (float*)malloc(param_size * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < input_size; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Values between -1 and 1
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
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    cudaMalloc(&d_gamma, param_size * sizeof(float));
    cudaMalloc(&d_beta, param_size * sizeof(float));
    cudaMalloc(&d_batch_mean, param_size * sizeof(float));
    cudaMalloc(&d_batch_var, param_size * sizeof(float));
    cudaMalloc(&d_running_mean, param_size * sizeof(float));
    cudaMalloc(&d_running_var, param_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_mean, h_running_mean, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_var, h_running_var, param_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch training mode kernel
    int threads = 256;
    int blocks = channels;
    int shared_mem_size = 2 * threads * sizeof(float);  // For sum and squared sum
    bool is_training = true;
    
    printf("Running Batch Normalization forward pass (training mode)...\n");
    
    cudaEventRecord(start);
    batch_norm_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        d_input, d_output, d_gamma, d_beta,
        d_batch_mean, d_batch_var, d_running_mean, d_running_var,
        batch_size, channels, spatial_size, 
        momentum, epsilon, is_training
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float training_time = 0;
    cudaEventElapsedTime(&training_time, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_batch_mean, d_batch_mean, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_batch_var, d_batch_var, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_running_mean, d_running_mean, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_running_var, d_running_var, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Launch inference mode kernel
    is_training = false;
    
    printf("Running Batch Normalization forward pass (inference mode)...\n");
    
    cudaEventRecord(start);
    batch_norm_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        d_input, d_output, d_gamma, d_beta,
        d_batch_mean, d_batch_var, d_running_mean, d_running_var,
        batch_size, channels, spatial_size, 
        momentum, epsilon, is_training
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float inference_time = 0;
    cudaEventElapsedTime(&inference_time, start, stop);
    
    // Verification
    printf("\nTraining mode time: %.3f ms\n", training_time);
    printf("Inference mode time: %.3f ms\n", inference_time);
    
    // Print some statistics
    printf("\nStatistics for first few channels:\n");
    for (int c = 0; c < 3 && c < channels; c++) {
        printf("Channel %d: Batch Mean=%.6f, Batch Var=%.6f, Running Mean=%.6f, Running Var=%.6f\n",
               c, h_batch_mean[c], h_batch_var[c], h_running_mean[c], h_running_var[c]);
    }
    
    // Compute output statistics for verification
    printf("\nOutput statistics for first channel:\n");
    float out_mean = 0.0f;
    float out_var = 0.0f;
    
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = ((n * channels) + 0) * spatial_size + s;  // First channel
            out_mean += h_output[idx];
        }
    }
    out_mean /= (batch_size * spatial_size);
    
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial_size; s++) {
            int idx = ((n * channels) + 0) * spatial_size + s;  // First channel
            float diff = h_output[idx] - out_mean;
            out_var += diff * diff;
        }
    }
    out_var /= (batch_size * spatial_size);
    
    printf("Output Mean=%.6f, Output Var=%.6f\n", out_mean, out_var);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_batch_mean);
    cudaFree(d_batch_var);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    
    free(h_input);
    free(h_output);
    free(h_gamma);
    free(h_beta);
    free(h_batch_mean);
    free(h_batch_var);
    free(h_running_mean);
    free(h_running_var);
    
    return 0;
}