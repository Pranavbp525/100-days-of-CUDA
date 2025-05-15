#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple softmax backward kernel (one thread per sample)
__global__ void simple_softmax_backward_kernel(
    float* grad_input,    // Output: gradient w.r.t input
    const float* grad_output,  // Input: gradient w.r.t output
    const float* softmax_output,  // Input: softmax output from forward pass
    int batch_size,
    int feature_dim
) {
    // Each thread handles one sample
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample_idx < batch_size) {
        // Get pointers to this sample's data
        const float* sample_grad_output = grad_output + sample_idx * feature_dim;
        const float* sample_softmax_output = softmax_output + sample_idx * feature_dim;
        float* sample_grad_input = grad_input + sample_idx * feature_dim;
        
        // Step 1: Compute dot product of softmax_output and grad_output
        float dot_product = 0.0f;
        for (int i = 0; i < feature_dim; i++) {
            dot_product += sample_softmax_output[i] * sample_grad_output[i];
        }
        
        // Step 2: Compute gradient using the formula:
        // grad_input = softmax_output * (grad_output - dot_product)
        for (int i = 0; i < feature_dim; i++) {
            sample_grad_input[i] = sample_softmax_output[i] * 
                                 (sample_grad_output[i] - dot_product);
        }
    }
}

int main() {
    // Configuration
    int batch_size = 10;
    int feature_dim = 100;
    int total_size = batch_size * feature_dim;
    
    printf("Running softmax backward on %d samples with %d features each\n", batch_size, feature_dim);
    
    // Allocate host memory
    float *h_grad_output = (float*)malloc(total_size * sizeof(float));
    float *h_softmax_output = (float*)malloc(total_size * sizeof(float));
    float *h_grad_input = (float*)malloc(total_size * sizeof(float));
    
    // Initialize with test values
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        // Create a valid probability distribution for softmax_output
        for (int f = 0; f < feature_dim; f++) {
            h_softmax_output[b * feature_dim + f] = (float)rand() / RAND_MAX;
            sum += h_softmax_output[b * feature_dim + f];
        }
        // Normalize to make it a valid probability distribution
        for (int f = 0; f < feature_dim; f++) {
            h_softmax_output[b * feature_dim + f] /= sum;
        }
        
        // Random gradients for grad_output
        for (int f = 0; f < feature_dim; f++) {
            h_grad_output[b * feature_dim + f] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    
    // Allocate device memory
    float *d_grad_output, *d_softmax_output, *d_grad_input;
    cudaMalloc(&d_grad_output, total_size * sizeof(float));
    cudaMalloc(&d_softmax_output, total_size * sizeof(float));
    cudaMalloc(&d_grad_input, total_size * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_grad_output, h_grad_output, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_softmax_output, h_softmax_output, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with one thread per sample
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    simple_softmax_backward_kernel<<<blocks, threads_per_block>>>(
        d_grad_input, d_grad_output, d_softmax_output, batch_size, feature_dim);
    
    // Copy result back to host
    cudaMemcpy(h_grad_input, d_grad_input, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify a few results (just print some values)
    printf("\nVerification (first few values of first sample):\n");
    for (int f = 0; f < 5 && f < feature_dim; f++) {
        printf("softmax_output[0,%d] = %.6f, grad_output[0,%d] = %.6f, grad_input[0,%d] = %.6f\n", 
               f, h_softmax_output[f], f, h_grad_output[f], f, h_grad_input[f]);
    }
    
    // Free memory
    free(h_grad_output);
    free(h_softmax_output);
    free(h_grad_input);
    cudaFree(d_grad_output);
    cudaFree(d_softmax_output);
    cudaFree(d_grad_input);
    
    return 0;
}