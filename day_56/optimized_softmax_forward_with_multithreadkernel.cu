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

// Kernel 2: Compute exp(x - max) for each element
__global__ void exp_kernel(float* input, float* exp_values, float* max_vals, int batch_size, int feature_dim) {
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointers to data
        float* sample = input + batch_idx * feature_dim;
        float* exp_out = exp_values + batch_idx * feature_dim;
        float max_val = max_vals[batch_idx];
        
        // Each thread computes exp(x - max) for its assigned elements
        for (int i = tid; i < feature_dim; i += blockDim.x) {
            exp_out[i] = expf(sample[i] - max_val);
        }
    }
}

// Kernel 3: Compute sum of exponentials for each sample
__global__ void sum_kernel(float* exp_values, float* sum_vals, int batch_size, int feature_dim) {
    extern __shared__ float shared_data[];
    
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointer to this sample's exponential values
        float* exp_sample = exp_values + batch_idx * feature_dim;
        
        // Each thread computes partial sum for its elements
        float thread_sum = 0.0f;
        for (int i = tid; i < feature_dim; i += blockDim.x) {
            thread_sum += exp_sample[i];
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

// Kernel 4: Normalize by dividing each element by the sum
__global__ void normalize_kernel(float* exp_values, float* output, float* sum_vals, int batch_size, int feature_dim) {
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointers to data
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
    int batch_size = 1000;
    int feature_dim = 1000;
    int total_size = batch_size * feature_dim;
    
    printf("Running optimized softmax on %d samples with %d features each\n", batch_size, feature_dim);
    
    // Allocate host memory
    float *h_input = (float*)malloc(total_size * sizeof(float));
    float *h_output = (float*)malloc(total_size * sizeof(float));
    
    // Initialize input with test values
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < feature_dim; f++) {
            // Simple pattern: increasing values with a spike
            h_input[b * feature_dim + f] = (float)f / feature_dim;
            if (f == b % feature_dim) {
                h_input[b * feature_dim + f] += 3.0f;  // Make this value stand out
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
    
    // Launch kernels in sequence
    find_max_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_input, d_max_vals, batch_size, feature_dim);
    exp_kernel<<<blocks, threads_per_block>>>(d_input, d_exp_values, d_max_vals, batch_size, feature_dim);
    sum_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_exp_values, d_sum_vals, batch_size, feature_dim);
    normalize_kernel<<<blocks, threads_per_block>>>(d_exp_values, d_output, d_sum_vals, batch_size, feature_dim);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify a few results
    printf("\nExecution time: %.3f ms\n", milliseconds);
    printf("\nVerifying results (checking sums equal 1.0):\n");
    
    bool all_correct = true;
    for (int b = 0; b < 3; b++) {  // Check first 3 samples
        float sum = 0.0f;
        int largest_idx = -1;
        float largest_val = -1.0f;
        
        for (int f = 0; f < feature_dim; f++) {
            float val = h_output[b * feature_dim + f];
            sum += val;
            if (val > largest_val) {
                largest_val = val;
                largest_idx = f;
            }
        }
        
        printf("Sample %d: Sum = %.6f, Largest probability: %.6f at index %d\n", 
               b, sum, largest_val, largest_idx);
        
        if (fabsf(sum - 1.0f) > 1e-5) {
            all_correct = false;
            printf("  ERROR: Sum should be 1.0!\n");
        }
        
        // Verify that the spike in the input resulted in the highest probability
        if (largest_idx != b % feature_dim) {
            printf("  NOTE: Largest probability not at expected index!\n");
        }
    }
    
    if (all_correct) {
        printf("\nAll checked samples have correct sums of 1.0\n");
    }
    
    // Compare to baseline single-threaded implementation (for educational purposes)
    printf("\nCompared to baseline single-threaded approach, this implementation offers:\n");
    printf("1. Better parallelism: Using %d threads per sample instead of 1\n", threads_per_block);
    printf("2. Logarithmic reductions: O(log n) complexity for finding max and sum\n");
    printf("3. Specialized kernels: Each kernel optimized for its specific task\n");
    printf("4. Improved memory access: Coalesced access patterns within each kernel\n");
    
    // Clean up
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