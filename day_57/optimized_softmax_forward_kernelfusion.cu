#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel 1: Find maximum value for each sample
__global__ void find_max_kernel(float* input, float* max_vals, int batch_size, int feature_dim) {
    extern __shared__ float shared_data[];
    
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointer to this sample's data
        float* sample = input + batch_idx * feature_dim;
        
        // Each thread finds max value in its assigned range
        float thread_max = -INFINITY;
        for (int i = tid; i < feature_dim; i += blockDim.x) {
            thread_max = fmaxf(thread_max, sample[i]);
        }
        
        // Store thread's max in shared memory
        shared_data[tid] = thread_max;
        __syncthreads();
        
        // Parallel reduction to find max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
            }
            __syncthreads();
        }
        
        // Thread 0 writes the result to global memory
        if (tid == 0) {
            max_vals[batch_idx] = shared_data[0];
        }
    }
}

// Kernel 2: Compute exp and sum in a single kernel
__global__ void compute_exp_sum_kernel(float* input, float* exp_values, float* max_vals, 
                                     float* sum_vals, int batch_size, int feature_dim) {
    extern __shared__ float shared_data[];
    
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointers and values
        float* sample = input + batch_idx * feature_dim;
        float* exp_out = exp_values + batch_idx * feature_dim;
        float max_val = max_vals[batch_idx];
        
        // Each thread computes exp and tracks partial sum
        float thread_sum = 0.0f;
        for (int i = tid; i < feature_dim; i += blockDim.x) {
            float exp_val = expf(sample[i] - max_val);
            exp_out[i] = exp_val;
            thread_sum += exp_val;
        }
        
        // Store thread's sum in shared memory
        shared_data[tid] = thread_sum;
        __syncthreads();
        
        // Parallel reduction to compute total sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }
        
        // Thread 0 writes the result to global memory
        if (tid == 0) {
            sum_vals[batch_idx] = shared_data[0];
        }
    }
}

// Kernel 3: Normalize by dividing each element by the sum
__global__ void normalize_kernel(float* exp_values, float* output, float* sum_vals, 
                               int batch_size, int feature_dim) {
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointers and values
        float* exp_sample = exp_values + batch_idx * feature_dim;
        float* out_sample = output + batch_idx * feature_dim;
        float sum = sum_vals[batch_idx];
        
        // Each thread normalizes its assigned elements
        for (int i = tid; i < feature_dim; i += blockDim.x) {
            out_sample[i] = exp_sample[i] / sum;
        }
    }
}

int main() {
    // Configuration
    int batch_size = 100;
    int feature_dim = 1000;
    int total_size = batch_size * feature_dim;
    
    printf("Softmax for %d samples with %d features each\n", batch_size, feature_dim);
    
    // Allocate host memory
    float *h_input = (float*)malloc(total_size * sizeof(float));
    float *h_output = (float*)malloc(total_size * sizeof(float));
    
    // Initialize input with some test values
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < feature_dim; f++) {
            h_input[b * feature_dim + f] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
            // Make one value stand out to easily verify softmax
            if (f == b % feature_dim) {
                h_input[b * feature_dim + f] += 5.0f;
            }
        }
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_max_vals, *d_exp_values, *d_sum_vals;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    cudaMalloc(&d_max_vals, batch_size * sizeof(float));
    cudaMalloc(&d_exp_values, total_size * sizeof(float));
    cudaMalloc(&d_sum_vals, batch_size * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks = batch_size;  // One block per sample
    int shared_mem_size = threads_per_block * sizeof(float);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start);
    
    // Launch kernels with balanced approach
    find_max_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input, d_max_vals, batch_size, feature_dim);
    
    compute_exp_sum_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input, d_exp_values, d_max_vals, d_sum_vals, batch_size, feature_dim);
    
    normalize_kernel<<<blocks, threads_per_block>>>(
        d_exp_values, d_output, d_sum_vals, batch_size, feature_dim);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Verify a few results
    printf("Softmax execution time: %.3f ms\n", milliseconds);
    
    // Check a few samples to see if their sums equal 1.0
    for (int b = 0; b < 3; b++) {  // Check first 3 samples
        float sum = 0.0f;
        for (int f = 0; f < feature_dim; f++) {
            sum += h_output[b * feature_dim + f];
        }
        printf("Sample %d sum: %.6f\n", b, sum);
    }
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max_vals);
    cudaFree(d_exp_values);
    cudaFree(d_sum_vals);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input);
    free(h_output);
    
    return 0;
}