#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Max Pooling Forward Pass Kernel
__global__ void max_pooling_forward_kernel(
    const float* input,     // Input tensor [N,C,H,W]
    float* output,          // Output tensor [N,C,H_out,W_out]
    int* indices,           // Indices for backward pass [N,C,H_out,W_out]
    int batch_size,         // N
    int channels,           // C
    int height,             // H
    int width,              // W
    int kernel_size,        // Pooling window size (assuming square)
    int stride,             // Stride (assuming same for H and W)
    int pad                 // Padding (assuming same for all sides)
) {
    // Calculate output dimensions
    int height_out = (height + 2 * pad - kernel_size) / stride + 1;
    int width_out = (width + 2 * pad - kernel_size) / stride + 1;
    
    // Calculate global position
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within bounds
    if (pos >= batch_size * channels * height_out * width_out) return;
    
    // Convert linear position to 4D coordinates [n,c,h,w]
    int w_out = pos % width_out;
    int h_out = (pos / width_out) % height_out;
    int c = (pos / (width_out * height_out)) % channels;
    int n = pos / (width_out * height_out * channels);
    
    // Calculate the window's top-left corner in the input
    int h_start = h_out * stride - pad;
    int w_start = w_out * stride - pad;
    
    // Calculate the window's bottom-right corner in the input
    int h_end = min(h_start + kernel_size, height);
    int w_end = min(w_start + kernel_size, width);
    
    // Adjust for padding
    h_start = max(0, h_start);
    w_start = max(0, w_start);
    
    // Find the maximum value in the window
    float max_val = -INFINITY;
    int max_idx = -1;
    
    // Input offset for this batch and channel
    int input_offset = ((n * channels + c) * height) * width;
    
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            int idx = input_offset + h * width + w;
            float val = input[idx];
            
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
        }
    }
    
    // Output offset
    int output_offset = ((n * channels + c) * height_out + h_out) * width_out + w_out;
    
    // Write output and save index for backward pass
    output[output_offset] = max_val;
    indices[output_offset] = max_idx;
}

int main() {
    // Input dimensions [N,C,H,W]
    int batch_size = 16;
    int channels = 64;
    int height = 28;
    int width = 28;
    
    // Pooling parameters
    int kernel_size = 2;
    int stride = 2;
    int pad = 0;
    
    // Calculate output dimensions
    int height_out = (height + 2 * pad - kernel_size) / stride + 1;
    int width_out = (width + 2 * pad - kernel_size) / stride + 1;
    
    // Allocate host memory
    size_t input_size = batch_size * channels * height * width;
    size_t output_size = batch_size * channels * height_out * width_out;
    
    float* h_input = (float*)malloc(input_size * sizeof(float));
    float* h_output = (float*)malloc(output_size * sizeof(float));
    int* h_indices = (int*)malloc(output_size * sizeof(int));
    
    // Initialize input with random values
    for (size_t i = 0; i < input_size; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_indices, output_size * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks_forward = (output_size + threads_per_block - 1) / threads_per_block;
    
    // Launch max pooling forward kernel
    printf("Running max pooling forward pass...\n");
    max_pooling_forward_kernel<<<blocks_forward, threads_per_block>>>(
        d_input, d_output, d_indices,
        batch_size, channels, height, width,
        kernel_size, stride, pad);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, output_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Optionally print a small part of the output for verification
    printf("First 10 output values after max pooling:\n");
    for (int i = 0; i < 10 && i < output_size; ++i) {
        printf("%f (idx %d) ", h_output[i], h_indices[i]);
    }
    printf("\n");
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
    free(h_input);
    free(h_output);
    free(h_indices);
    
    printf("Max pooling forward operation completed successfully!\n");
    return 0;
}

