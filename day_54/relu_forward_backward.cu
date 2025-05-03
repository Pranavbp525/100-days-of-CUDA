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

// Timer macro for performance measurement
#define TIME_FUNC(func, time_var) \
do { \
    cudaEvent_t start, stop; \
    CUDA_CHECK(cudaEventCreate(&start)); \
    CUDA_CHECK(cudaEventCreate(&stop)); \
    CUDA_CHECK(cudaEventRecord(start)); \
    func; \
    CUDA_CHECK(cudaEventRecord(stop)); \
    CUDA_CHECK(cudaEventSynchronize(stop)); \
    CUDA_CHECK(cudaEventElapsedTime(&time_var, start, stop)); \
    CUDA_CHECK(cudaEventDestroy(start)); \
    CUDA_CHECK(cudaEventDestroy(stop)); \
    time_var /= 1000.0f; /* Convert to seconds */ \
} while(0)

// Forward pass: ReLU kernel
__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Backward pass: ReLU gradient kernel
__global__ void relu_backward_kernel(float* input, float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // ReLU gradient: 1 if input > 0, 0 otherwise
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0;
    }
}

// CPU implementation of ReLU forward pass
void relu_forward_cpu(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

// CPU implementation of ReLU backward pass
void relu_backward_cpu(float* input, float* grad_output, float* grad_input, int size) {
    for (int i = 0; i < size; i++) {
        grad_input[i] = (input[i] > 0) ? grad_output[i] : 0;
    }
}

// CUDA implementation of ReLU forward pass
void relu_forward_cuda(float* h_input, float* h_output, int size) {
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
    relu_forward_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

// CUDA implementation of ReLU backward pass
void relu_backward_cuda(float* h_input, float* h_grad_output, float* h_grad_input, int size) {
    // Device memory pointers
    float *d_input, *d_grad_output, *d_grad_input;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_output, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_grad_input, size * sizeof(float)));
    
    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Launch kernel
    relu_backward_kernel<<<numBlocks, blockSize>>>(d_input, d_grad_output, d_grad_input, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_grad_input));
}

// Function to verify results
bool verify_results(float* a, float* b, int size, const char* op_name) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            printf("Verification failed for %s at index %d: CPU = %f, GPU = %f\n", 
                   op_name, i, a[i], b[i]);
            return false;
        }
    }
    printf("Verification successful for %s: CPU and GPU outputs match!\n", op_name);
    return true;
}

// Function to print a few samples for visual inspection
void print_samples(float* arr, int size, const char* name) {
    printf("%s (first 5 elements): ", name);
    for (int i = 0; i < 5 && i < size; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main() {
    // Set the size of the array (using a large size to see performance benefits)
    int size = 1 << 24;  // 16M elements
    printf("Processing array of size %d\n", size);
    
    // Allocate host memory
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output_cpu = (float*)malloc(size * sizeof(float));
    float *h_output_gpu = (float*)malloc(size * sizeof(float));
    float *h_grad_output = (float*)malloc(size * sizeof(float));
    float *h_grad_input_cpu = (float*)malloc(size * sizeof(float));
    float *h_grad_input_gpu = (float*)malloc(size * sizeof(float));
    
    // Initialize input data with random values
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        // Generate values between -10 and 10
        h_input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        
        // Random gradients for backward pass
        h_grad_output[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Print some sample input values
    print_samples(h_input, size, "Input");
    print_samples(h_grad_output, size, "Gradient Output");
    
    // =====================
    // Forward Pass Testing
    // =====================
    printf("\n--- Forward Pass Testing ---\n");
    
    // CPU implementation timing
    float cpu_forward_time;
    TIME_FUNC(relu_forward_cpu(h_input, h_output_cpu, size), cpu_forward_time);
    
    // GPU implementation timing
    float gpu_forward_time;
    TIME_FUNC(relu_forward_cuda(h_input, h_output_gpu, size), gpu_forward_time);
    
    // Verify forward pass results
    verify_results(h_output_cpu, h_output_gpu, size, "Forward Pass");
    
    // Print some sample output values
    print_samples(h_output_cpu, size, "CPU Forward Output");
    print_samples(h_output_gpu, size, "GPU Forward Output");
    
    // Print forward pass timing results
    printf("CPU forward time: %f seconds\n", cpu_forward_time);
    printf("GPU forward time: %f seconds\n", gpu_forward_time);
    printf("Forward speedup: %fx\n", cpu_forward_time / gpu_forward_time);
    
    // =====================
    // Backward Pass Testing
    // =====================
    printf("\n--- Backward Pass Testing ---\n");
    
    // CPU implementation timing
    float cpu_backward_time;
    TIME_FUNC(relu_backward_cpu(h_input, h_grad_output, h_grad_input_cpu, size), cpu_backward_time);
    
    // GPU implementation timing
    float gpu_backward_time;
    TIME_FUNC(relu_backward_cuda(h_input, h_grad_output, h_grad_input_gpu, size), gpu_backward_time);
    
    // Verify backward pass results
    verify_results(h_grad_input_cpu, h_grad_input_gpu, size, "Backward Pass");
    
    // Print some sample gradient values
    print_samples(h_grad_input_cpu, size, "CPU Backward Output");
    print_samples(h_grad_input_gpu, size, "GPU Backward Output");
    
    // Print backward pass timing results
    printf("CPU backward time: %f seconds\n", cpu_backward_time);
    printf("GPU backward time: %f seconds\n", gpu_backward_time);
    printf("Backward speedup: %fx\n", cpu_backward_time / gpu_backward_time);
    
    // Free host memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    free(h_grad_output);
    free(h_grad_input_cpu);
    free(h_grad_input_gpu);
    
    printf("\nAll tests completed successfully!\n");
    
    return 0;
}