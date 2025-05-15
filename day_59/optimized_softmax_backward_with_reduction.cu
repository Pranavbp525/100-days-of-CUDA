#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Optimized softmax backward kernel (multiple threads per sample)
__global__ void optimized_softmax_backward_kernel(
    float* grad_input,
    const float* grad_output,
    const float* softmax_output,
    int batch_size,
    int feature_dim
) {
    extern __shared__ float shared_data[];
    
    // Each block handles one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointers to this sample's data
        const float* sample_grad_output = grad_output + batch_idx * feature_dim;
        const float* sample_softmax_output = softmax_output + batch_idx * feature_dim;
        float* sample_grad_input = grad_input + batch_idx * feature_dim;
        
        // Step 1: Compute dot product using parallel reduction
        float thread_dot_product = 0.0f;
        for (int i = tid; i < feature_dim; i += blockDim.x) {
            thread_dot_product += sample_softmax_output[i] * sample_grad_output[i];
        }
        
        // Store thread's partial dot product in shared memory
        shared_data[tid] = thread_dot_product;
        __syncthreads();
        
        // Parallel reduction to compute the total dot product
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }
        
        // All threads now have access to the dot product in shared_data[0]
        float dot_product = shared_data[0];
        
        // Step 2: Each thread computes gradient for its assigned elements
        for (int i = tid; i < feature_dim; i += blockDim.x) {
            sample_grad_input[i] = sample_softmax_output[i] * 
                                 (sample_grad_output[i] - dot_product);
        }
    }
}

int main() {
    // Configuration
    int batch_size = 1000;
    int feature_dim = 1000;
    int total_size = batch_size * feature_dim;
    
    printf("Running softmax backward on %d samples with %d features each\n", batch_size, feature_dim);
    
    // Allocate host memory
    float *h_grad_output = (float*)malloc(total_size * sizeof(float));
    float *h_softmax_output = (float*)malloc(total_size * sizeof(float));
    float *h_grad_input = (float*)malloc(total_size * sizeof(float));
    
    // Initialize with test values
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        // Initialize softmax_output as a valid probability distribution
        for (int f = 0; f < feature_dim; f++) {
            h_softmax_output[b * feature_dim + f] = (float)rand() / RAND_MAX;
            sum += h_softmax_output[b * feature_dim + f];
        }
        // Normalize to ensure it's a proper probability distribution
        for (int f = 0; f < feature_dim; f++) {
            h_softmax_output[b * feature_dim + f] /= sum;
        }
        
        // Random values for grad_output
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
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel with each block handling one sample
    int threads_per_block = 256;
    int blocks = batch_size;  // One block per sample
    int shared_mem_size = threads_per_block * sizeof(float);  // For dot product reduction
    
    // Record start time
    cudaEventRecord(start);
    
    optimized_softmax_backward_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_grad_input, d_grad_output, d_softmax_output, batch_size, feature_dim);
    
    // Record end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_grad_input, d_grad_input, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing information
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    // Verify a few results (just print some values)
    printf("\nResults for first sample:\n");
    printf("Dot product: %.6f\n", 
        h_grad_input[0] / h_softmax_output[0] + h_grad_output[0]);  // Back-calculate the dot product
    
    printf("Some output values:\n");
    for (int f = 0; f < 5 && f < feature_dim; f++) {
        printf("[%d] softmax: %.6f, grad_output: %.6f, grad_input: %.6f\n", 
               f, h_softmax_output[f], h_grad_output[f], h_grad_input[f]);
    }
    
    // Free memory
    free(h_grad_output);
    free(h_softmax_output);
    free(h_grad_input);
    cudaFree(d_grad_output);
    cudaFree(d_softmax_output);
    cudaFree(d_grad_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}