#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Simple Cross-Entropy Loss Forward Pass
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
    float *d_predictions, *d_losses;
    int *d_targets;
    
    cudaMalloc(&d_predictions, predictions_size * sizeof(float));
    cudaMalloc(&d_targets, targets_size * sizeof(int));
    cudaMalloc(&d_losses, targets_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_predictions, h_predictions, predictions_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, targets_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    printf("Computing Cross-Entropy Loss for %d samples with %d classes each\n", batch_size, num_classes);
    
    cudaEventRecord(start);
    cross_entropy_forward_kernel<<<blocks, threads_per_block>>>(
        d_predictions, d_targets, d_losses, batch_size, num_classes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back to host
    cudaMemcpy(h_losses, d_losses, targets_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float total_loss = 0.0f;
    float min_loss = h_losses[0];
    float max_loss = h_losses[0];
    
    for (int n = 0; n < batch_size; n++) {
        total_loss += h_losses[n];
        min_loss = fminf(min_loss, h_losses[n]);
        max_loss = fmaxf(max_loss, h_losses[n]);
    }
    
    float avg_loss = total_loss / batch_size;
    
    // Display results
    printf("\nResults:\n");
    printf("Execution time: %.3f ms\n", milliseconds);
    printf("Average loss: %.4f\n", avg_loss);
    printf("Min loss: %.4f\n", min_loss);
    printf("Max loss: %.4f\n", max_loss);
    
    // Show some examples
    printf("\nFirst 5 samples:\n");
    printf("Sample | Target | Prediction | Loss\n");
    printf("-------|--------|------------|------\n");
    for (int n = 0; n < 5; n++) {
        int target = h_targets[n];
        float pred = h_predictions[n * num_classes + target];
        printf("   %d   |   %d    |   %.4f   | %.4f\n", n, target, pred, h_losses[n]);
    }
    
    // Clean up
    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_losses);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_predictions);
    free(h_targets);
    free(h_losses);
    
    return 0;
}