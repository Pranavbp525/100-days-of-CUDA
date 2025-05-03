#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ReLU kernel
__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CPU implementation for comparison
void relu_cpu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

// CUDA implementation
void relu_cuda(float* h_input, float* h_output, int size) {
    // Device memory pointers
    float *d_input, *d_output;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size * sizeof(float)));
    
    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Launch kernel
    relu_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// Function to verify results
bool verify_results(float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            printf("Verification failed at index %d: CPU = %f, GPU = %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Set the size of the array
    int size = 1 << 24;  // 16M elements
    printf("Processing array of size %d\n", size);
    
    // Allocate host memory
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output_cpu = (float*)malloc(size * sizeof(float));
    float *h_output_gpu = (float*)malloc(size * sizeof(float));
    
    // Initialize input data with some random values
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        // Generate values between -10 and 10
        h_input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }
    
    // CPU implementation timing
    clock_t cpu_start = clock();
    relu_cpu(h_input, h_output_cpu, size);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    // GPU implementation timing
    clock_t gpu_start = clock();
    relu_cuda(h_input, h_output_gpu, size);
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;
    
    // Verify results
    bool success = verify_results(h_output_cpu, h_output_gpu, size);
    if (success) {
        printf("Results verified: CPU and GPU outputs match!\n");
    }
    
    // Print timing results
    printf("CPU time: %f seconds\n", cpu_time);
    printf("GPU time: %f seconds\n", gpu_time);
    printf("Speedup: %fx\n", cpu_time / gpu_time);
    
    // Free host memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    
    return 0;
}