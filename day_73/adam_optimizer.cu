#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Adam optimizer update kernel
__global__ void adam_update_kernel(
    float* params,             // Model parameters to update
    const float* gradients,    // Gradients of parameters
    float* m,                  // First moment (momentum)
    float* v,                  // Second moment (velocity)
    int size,                  // Number of parameters
    float learning_rate,       // Learning rate (alpha)
    float beta1,               // Exponential decay rate for 1st moment
    float beta2,               // Exponential decay rate for 2nd moment
    float epsilon,             // Small constant for numerical stability
    int t                      // Current timestep (for bias correction)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Get current gradient
        float g = gradients[idx];
        
        // Update biased first moment estimate (momentum)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        
        // Update biased second moment estimate (velocity)
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - pow(beta1, t));
        
        // Compute bias-corrected second moment estimate
        float v_hat = v[idx] / (1.0f - pow(beta2, t));
        
        // Update parameters
        params[idx] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    }
}

int main() {
    // Example settings
    int param_size = 10000000;  // Number of parameters (e.g., a large model)
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    int num_iterations = 10;
    
    printf("Adam Optimizer CUDA Implementation\n");
    printf("Parameters: %d\n", param_size);
    printf("Learning rate: %f\n", learning_rate);
    printf("Beta1: %f\n", beta1);
    printf("Beta2: %f\n", beta2);
    printf("Epsilon: %e\n", epsilon);
    printf("Iterations: %d\n\n", num_iterations);
    
    // Allocate host memory
    float *h_params = (float*)malloc(param_size * sizeof(float));
    float *h_gradients = (float*)malloc(param_size * sizeof(float));
    float *h_m = (float*)malloc(param_size * sizeof(float));
    float *h_v = (float*)malloc(param_size * sizeof(float));
    
    // Initialize parameters and gradients
    for (int i = 0; i < param_size; i++) {
        h_params[i] = ((float)rand() / RAND_MAX) * 0.1f;  // Small random values
        h_gradients[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;  // Small random gradients
        h_m[i] = 0.0f;  // Initialize first moment to zero
        h_v[i] = 0.0f;  // Initialize second moment to zero
    }
    
    // Sample some values before update
    printf("Initial values (first 5 parameters):\n");
    for (int i = 0; i < 5; i++) {
        printf("Param[%d] = %.6f, Gradient[%d] = %.6f\n", i, h_params[i], i, h_gradients[i]);
    }
    
    // Allocate device memory
    float *d_params, *d_gradients, *d_m, *d_v;
    cudaMalloc(&d_params, param_size * sizeof(float));
    cudaMalloc(&d_gradients, param_size * sizeof(float));
    cudaMalloc(&d_m, param_size * sizeof(float));
    cudaMalloc(&d_v, param_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_params, h_params, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, h_gradients, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, param_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, param_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Configure kernel launch parameters
    int threads_per_block = 256;
    int blocks = (param_size + threads_per_block - 1) / threads_per_block;
    
    // Record start time
    cudaEventRecord(start);
    
    // Run optimizer for several iterations
    for (int iter = 1; iter <= num_iterations; iter++) {
        adam_update_kernel<<<blocks, threads_per_block>>>(
            d_params, d_gradients, d_m, d_v,
            param_size, learning_rate, beta1, beta2, epsilon, iter
        );
        
        // In a real scenario, we would recompute gradients after each update
        // For simplicity, we're just reusing the same gradients here
    }
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Copy updated parameters back to host
    cudaMemcpy(h_params, d_params, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, param_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Display results
    printf("\nAfter %d iterations (%.2f ms):\n", num_iterations, milliseconds);
    for (int i = 0; i < 5; i++) {
        printf("Param[%d] = %.6f, m[%d] = %.6f, v[%d] = %.6f\n", 
               i, h_params[i], i, h_m[i], i, h_v[i]);
    }
    
    // Calculate average update magnitude
    float total_update = 0.0f;
    for (int i = 0; i < param_size; i++) {
        total_update += fabs(h_params[i] - h_gradients[i] * learning_rate * num_iterations);
    }
    
    printf("\nAverage parameter change: %.6f\n", total_update / param_size);
    printf("Processing rate: %.2f million parameters/second\n", 
           (param_size * num_iterations) / (milliseconds * 1000.0f));
    
    // Clean up
    cudaFree(d_params);
    cudaFree(d_gradients);
    cudaFree(d_m);
    cudaFree(d_v);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_params);
    free(h_gradients);
    free(h_m);
    free(h_v);
    
    return 0;
}