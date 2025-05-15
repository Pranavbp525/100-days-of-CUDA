#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Average Pooling Forward Pass Kernel
__global__ void avg_pooling_forward_kernel(
    const float* input,     // Input tensor [N,C,H,W]
    float* output,          // Output tensor [N,C,H_out,W_out]
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
    
    // Compute the average value in the window
    float sum = 0.0f;
    int count = 0;
    
    // Input offset for this batch and channel
    int input_offset = ((n * channels + c) * height) * width;
    
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            sum += input[input_offset + h * width + w];
            count++;
        }
    }
    
    // Output offset
    int output_offset = ((n * channels + c) * height_out + h_out) * width_out + w_out;
    
    // Write average to output
    output[output_offset] = count > 0 ? sum / count : 0.0f;
}

// Average Pooling Backward Pass Kernel
__global__ void avg_pooling_backward_kernel(
    float* grad_input,        // Gradient w.r.t. input [N,C,H,W]
    const float* grad_output, // Gradient w.r.t. output [N,C,H_out,W_out]
    int batch_size,          // N
    int channels,            // C
    int height,              // H
    int width,               // W
    int height_out,          // H_out
    int width_out,           // W_out
    int kernel_size,         // Pooling window size
    int stride,              // Stride
    int pad                  // Padding
) {
    // Calculate global position
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within bounds
    if (pos >= batch_size * channels * height_out * width_out) return;
    
    // Convert linear position to 4D coordinates [n,c,h,w]
    int w_out = pos % width_out;
    int h_out = (pos / width_out) % height_out;
    int c = (pos / (width_out * height_out)) % channels;
    int n = pos / (width_out * height_out * channels);
    
    // Gradient from this output position
    float grad = grad_output[pos];
    
    // Calculate the window's top-left corner in the input
    int h_start = h_out * stride - pad;
    int w_start = w_out * stride - pad;
    
    // Calculate the window's bottom-right corner in the input
    int h_end = min(h_start + kernel_size, height);
    int w_end = min(w_start + kernel_size, width);
    
    // Adjust for padding
    h_start = max(0, h_start);
    w_start = max(0, w_start);
    
    // Count valid elements in the window
    int count = (h_end - h_start) * (w_end - w_start);
    
    // Evenly distribute the gradient to all input elements in the window
    float grad_per_element = count > 0 ? grad / count : 0.0f;
    
    // Input gradient offset for this batch and channel
    int input_offset = ((n * channels + c) * height) * width;
    
    for (int h = h_start; h < h_end; h++) {
        for (int w = w_start; w < w_end; w++) {
            int idx = input_offset + h * width + w;
            atomicAdd(&grad_input[idx], grad_per_element);
        }
    }
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
    float* h_grad_input = (float*)malloc(input_size * sizeof(float));
    
    // Initialize input with random values
    for (size_t i = 0; i < input_size; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
        h_grad_input[i] = 0.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_grad_input;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grad_input, 0, input_size * sizeof(float));
    
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks = (output_size + threads_per_block - 1) / threads_per_block;
    
    // Run forward pass
    printf("Running average pooling forward pass...\n");
    avg_pooling_forward_kernel<<<blocks, threads_per_block>>>(
        d_input, d_output,
        batch_size, channels, height, width,
        kernel_size, stride, pad);
    cudaDeviceSynchronize();
    
    // Use the output of the forward pass as grad_output for backward pass (for test)
    printf("Running average pooling backward pass...\n");
    avg_pooling_backward_kernel<<<blocks, threads_per_block>>>(
        d_grad_input, d_output,
        batch_size, channels, height, width,
        height_out, width_out,
        kernel_size, stride, pad);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_input, d_grad_input, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Optionally print a small part of the output for verification
    printf("First 10 output values after avg pooling forward:\n");
    for (int i = 0; i < 10 && i < output_size; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    printf("First 10 grad_input values after avg pooling backward:\n");
    for (int i = 0; i < 10 && i < input_size; ++i) {
        printf("%f ", h_grad_input[i]);
    }
    printf("\n");
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_input);
    free(h_input);
    free(h_output);
    free(h_grad_input);
    
    printf("Avg pooling forward and backward operations completed successfully!\n");
    return 0;
}