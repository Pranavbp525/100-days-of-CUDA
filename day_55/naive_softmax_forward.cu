#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void softmax_kernel(float* input, float* output, int batch_size, int feature_dim) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample_idx < batch_size) {
        float* sample_in = input + sample_idx * feature_dim;
        float* sample_out = output + sample_idx * feature_dim;
        
        float max_val = sample_in[0];
        for (int i = 1; i < feature_dim; i++) {
            if (sample_in[i] > max_val) {
                max_val = sample_in[i];
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < feature_dim; i++) {
            sample_out[i] = expf(sample_in[i] - max_val);
            sum += sample_out[i];
        }
        
        for (int i = 0; i < feature_dim; i++) {
            sample_out[i] /= sum;
        }
    }
}

int main() {
    int batch_size = 100;   
    int feature_dim = 10;   
    int total_size = batch_size * feature_dim;
    
    printf("Running softmax on %d samples with %d features each\n", batch_size, feature_dim);
    
    float *h_input = (float*)malloc(total_size * sizeof(float));
    float *h_output = (float*)malloc(total_size * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        for (int f = 0; f < feature_dim; f++) {
            h_input[b * feature_dim + f] = (float)f;
            if (f == b % feature_dim) {
                h_input[b * feature_dim + f] += 5.0f;  
            }
        }
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    softmax_kernel<<<blocks, threads_per_block>>>(d_input, d_output, batch_size, feature_dim);
    
    cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\nSample outputs (checking sums equal 1.0):\n");
    for (int b = 0; b < 3; b++) {  
        float sum = 0.0f;
        printf("Sample %d: [", b);
        for (int f = 0; f < feature_dim; f++) {
            float val = h_output[b * feature_dim + f];
            printf("%.4f", val);
            if (f < feature_dim - 1) printf(", ");
            sum += val;
        }
        printf("] Sum: %.6f\n", sum);
    }
    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}