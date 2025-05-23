#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Cross-Entropy Loss Forward Pass (from previous implementation)
__global__ void cross_entropy_forward_kernel(
    const float* predictions,    // Softmax predictions [N, C]
    const int* targets,          // Target class indices [N]
    float* losses,               // Per-sample losses [N]
    int batch_size,              // N
    int num_classes              // C
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < batch_size) {
        int target_class = targets[n];
        
        // Get prediction for the correct class
        float pred = predictions[n * num_classes + target_class];
        
        // Clamp prediction to avoid log(0)
        pred = fmaxf(pred, 1e-7f);
        
        // Compute cross-entropy loss: -log(prediction)
        losses[n] = -logf(pred);
    }
}

// Simple Cross-Entropy Backward Pass
__global__ void cross_entropy_backward_kernel(
    float* grad_predictions,     // Gradient w.r.t. predictions [N, C]
    const float* predictions,    // Softmax predictions [N, C]
    const int* targets,          // Target class indices [N]
    int batch_size,              // N
    int num_classes              // C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Convert linear index to 2D coordinates
    int n = idx / num_classes;  // Batch index
    int c = idx % num_classes;  // Class index
    
    if (n < batch_size && c < num_classes) {
        int target_class = targets[n];
        
        if (c == target_class) {
            // Gradient for the target class: -1/prediction
            float pred = fmaxf(predictions[idx], 1e-7f);  // Clamp to avoid division by 0
            grad_predictions[idx] = -1.0f / pred;
        } else {
            // Gradient for non-target classes: 0
            grad_predictions[idx] = 0.0f;
        }
        
        // Scale by batch size for averaging
        grad_predictions[idx] /= batch_size;
    }
}

int main() {
    // Test parameters
    int batch_size = 1000;      // N
    int num_classes = 10;       // C (e.g., MNIST classes)
    
    size_t predictions_size = batch_size * num_classes;
    size_t targets_size = batch_size;
    
    // Allocate host memory
    float *h_predictions = (float*)malloc(predictions_size * sizeof(float));
    int *h_targets = (int*)malloc(targets_size * sizeof(int));
    float *h_losses = (float*)malloc(targets_size * sizeof(float));
    float *h_grad_predictions = (float*)malloc(predictions_size * sizeof(float));
    
    // Initialize data with mock softmax predictions
    for (int n = 0; n < batch_size; n++) {
        // Random target class
        h_targets[n] = rand() % num_classes;
        
        // Create mock softmax predictions (sum to 1.0)
        float sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            h_predictions[n * num_classes + c] = (float)rand() / RAND_MAX;
            sum += h_predictions[n * num_classes + c];
        }
        
        // Normalize to create valid probabilities
        for (int c = 0; c < num_classes; c++) {
            h_predictions[n * num_classes + c] /= sum;
        }
    }
    
    // Allocate device memory
    float *d_predictions, *d_losses, *d_grad_predictions;
    int *d_targets;
    
    cudaMalloc(&d_predictions, predictions_size * sizeof(float));
    cudaMalloc(&d_targets, targets_size * sizeof(int));
    cudaMalloc(&d_losses, targets_size * sizeof(float));
    cudaMalloc(&d_grad_predictions, predictions_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_predictions, h_predictions, predictions_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, targets_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Cross-Entropy Loss Forward and Backward Pass\n");
    printf("Batch size: %d, Classes: %d\n\n", batch_size, num_classes);
    
    // Step 1: Forward pass
    int threads_per_block = 256;
    int blocks_forward = (batch_size + threads_per_block - 1) / threads_per_block;
    
    printf("Running forward pass...\n");
    cudaEventRecord(start);
    cross_entropy_forward_kernel<<<blocks_forward, threads_per_block>>>(
        d_predictions, d_targets, d_losses, batch_size, num_classes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float forward_time = 0;
    cudaEventElapsedTime(&forward_time, start, stop);
    
    // Step 2: Backward pass
    int total_elements = batch_size * num_classes;
    int blocks_backward = (total_elements + threads_per_block - 1) / threads_per_block;
    
    printf("Running backward pass...\n");
    cudaEventRecord(start);
    cross_entropy_backward_kernel<<<blocks_backward, threads_per_block>>>(
        d_grad_predictions, d_predictions, d_targets, batch_size, num_classes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float backward_time = 0;
    cudaEventElapsedTime(&backward_time, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_losses, d_losses, targets_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_predictions, d_grad_predictions, predictions_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate average loss
    float total_loss = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        total_loss += h_losses[n];
    }
    float avg_loss = total_loss / batch_size;
    
    // Display timing results
    printf("\nResults:\n");
    printf("Forward pass time: %.3f ms\n", forward_time);
    printf("Backward pass time: %.3f ms\n", backward_time);
    printf("Average loss: %.4f\n\n", avg_loss);
    
    // Show sample gradients for first sample
    printf("Sample gradients for first sample:\n");
    printf("Target class: %d\n", h_targets[0]);
    printf("Class | Prediction | Gradient\n");
    printf("------|------------|----------\n");
    for (int c = 0; c < num_classes; c++) {
        int idx = 0 * num_classes + c;  // First sample
        printf("  %d   |   %.4f   |  %.4f\n", 
               c, h_predictions[idx], h_grad_predictions[idx]);
    }
    
    // Count non-zero gradients (should equal batch_size)
    int non_zero_grads = 0;
    for (int i = 0; i < predictions_size; i++) {
        if (fabs(h_grad_predictions[i]) > 1e-6) {
            non_zero_grads++;
        }
    }
    printf("\nGradient verification:\n");
    printf("Non-zero gradients: %d (should be %d)\n", non_zero_grads, batch_size);
    
    // Clean up
    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_losses);
    cudaFree(d_grad_predictions);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_predictions);
    free(h_targets);
    free(h_losses);
    free(h_grad_predictions);
    
    return 0;
}