#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Dropout Forward Pass Kernel
__global__ void dropout_forward_kernel(
    float* output,           // Output tensor (same shape as input)
    const float* input,      // Input tensor
    float* mask,             // Binary mask for saving dropout pattern
    int size,                // Total number of elements
    float dropout_prob,      // Probability of dropping a unit
    unsigned long long seed, // Random seed
    bool is_training         // Whether we're in training mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        if (is_training) {
            // Initialize random state
            curandState state;
            curand_init(seed, idx, 0, &state);
            
            // Generate random number between 0 and 1
            float rand = curand_uniform(&state);
            
            // Create binary mask: 1 with probability (1-p), 0 with probability p
            mask[idx] = (rand > dropout_prob) ? 1.0f : 0.0f;
            
            // Apply mask and scale by 1/(1-p) to maintain expected sum
            float scale = 1.0f / (1.0f - dropout_prob);
            output[idx] = input[idx] * mask[idx] * scale;
        } else {
            // During inference we just pass the input through (already scaled properly during training)
            output[idx] = input[idx];
        }
    }
}

// Dropout Backward Pass Kernel
__global__ void dropout_backward_kernel(
    float* grad_input,         // Gradient w.r.t. input
    const float* grad_output,  // Gradient w.r.t. output
    const float* mask,         // Binary mask from forward pass
    int size,                  // Total number of elements
    float dropout_prob         // Probability of dropping a unit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Apply the same mask and scaling as in forward pass
        float scale = 1.0f / (1.0f - dropout_prob);
        grad_input[idx] = grad_output[idx] * mask[idx] * scale;
    }
}

int main() {
    // Set dimensions
    int batch_size = 256;
    int feature_dim = 1024;
    int total_size = batch_size * feature_dim;
    
    // Dropout parameters
    float dropout_prob = 0.5f;  // 50% dropout rate
    unsigned long long seed = 1234ULL;  // Random seed
    
    // Allocate host memory
    float *h_input = (float*)malloc(total_size * sizeof(float));
    float *h_output = (float*)malloc(total_size * sizeof(float));
    float *h_mask = (float*)malloc(total_size * sizeof(float));
    float *h_grad_output = (float*)malloc(total_size * sizeof(float));
    float *h_grad_input = (float*)malloc(total_size * sizeof(float));
    
    // Initialize input with some values
    for (int i = 0; i < total_size; i++) {
        h_input[i] = (float)i / total_size;  // Simple values for demonstration
    }
    
    // Initialize gradient with random values
    for (int i = 0; i < total_size; i++) {
        h_grad_output[i] = ((float)rand() / RAND_MAX) * 0.1f;  // Small random gradients
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_mask, *d_grad_output, *d_grad_input;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    cudaMalloc(&d_mask, total_size * sizeof(float));
    cudaMalloc(&d_grad_output, total_size * sizeof(float));
    cudaMalloc(&d_grad_input, total_size * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch forward kernel in training mode
    int threads_per_block = 256;
    int blocks = (total_size + threads_per_block - 1) / threads_per_block;
    bool is_training = true;
    
    printf("Running Dropout forward pass (training mode)...\n");
    
    cudaEventRecord(start);
    dropout_forward_kernel<<<blocks, threads_per_block>>>(
        d_output, d_input, d_mask, total_size, dropout_prob, seed, is_training
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float training_forward_time = 0;
    cudaEventElapsedTime(&training_forward_time, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask, d_mask, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Launch forward kernel in inference mode
    is_training = false;
    
    printf("Running Dropout forward pass (inference mode)...\n");
    
    cudaEventRecord(start);
    dropout_forward_kernel<<<blocks, threads_per_block>>>(
        d_output, d_input, d_mask, total_size, dropout_prob, seed, is_training
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float inference_forward_time = 0;
    cudaEventElapsedTime(&inference_forward_time, start, stop);
    
    // Launch backward kernel
    printf("Running Dropout backward pass...\n");
    
    cudaEventRecord(start);
    dropout_backward_kernel<<<blocks, threads_per_block>>>(
        d_grad_input, d_grad_output, d_mask, total_size, dropout_prob
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float backward_time = 0;
    cudaEventElapsedTime(&backward_time, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_grad_input, d_grad_input, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing info
    printf("\nTiming Results:\n");
    printf("Training forward pass: %.3f ms\n", training_forward_time);
    printf("Inference forward pass: %.3f ms\n", inference_forward_time);
    printf("Backward pass: %.3f ms\n", backward_time);
    
    // Calculate and print statistics
    printf("\nDropout Statistics:\n");
    
    // Count number of active (non-dropped) elements
    int active_count = 0;
    float sum_before = 0.0f;
    float sum_after = 0.0f;
    
    for (int i = 0; i < total_size; i++) {
        sum_before += h_input[i];
        sum_after += h_output[i];
        if (h_mask[i] > 0.0f) {
            active_count++;
        }
    }
    
    float dropout_rate = 1.0f - (float)active_count / total_size;
    printf("Applied dropout rate: %.2f%%\n", dropout_rate * 100.0f);
    printf("Expected dropout rate: %.2f%%\n", dropout_prob * 100.0f);
    printf("Sum before dropout: %.6f\n", sum_before);
    printf("Sum after dropout (training): %.6f\n", sum_after);
    printf("Ratio after/before: %.6f\n", sum_after / sum_before);
    
    // Print a few examples
    printf("\nSample values (first 10 elements):\n");
    printf("%-10s %-10s %-10s %-15s %-15s\n", "Index", "Input", "Mask", "Output", "Grad Input");
    for (int i = 0; i < 10; i++) {
        printf("%-10d %-10.4f %-10.4f %-15.4f %-15.4f\n", 
               i, h_input[i], h_mask[i], h_output[i], h_grad_input[i]);
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    
    free(h_input);
    free(h_output);
    free(h_mask);
    free(h_grad_output);
    free(h_grad_input);
    
    return 0;
}