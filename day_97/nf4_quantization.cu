#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>


#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


__constant__ float nf4_quantiles[16] = {
    -1.0000f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f,
    0.0911f, 0.1848f, 0.2844f, 0.3949f, 0.5251f, 0.6962f, 1.0000f, 0.0f // Note: 16th value is unused for 15 levels
};


__global__ void quantize_to_nf4_kernel(const float* input, uint8_t* output, float* d_scale, int block_size) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* s_block_data = shared_mem;
    
    float thread_max = 0.0f;
    for (int i = tid; i < block_size; i += blockDim.x) {
        s_block_data[i] = input[block_id * block_size + i];
        thread_max = fmaxf(thread_max, fabsf(s_block_data[i]));
    }
    __syncthreads();

    if (blockDim.x > 1) { 
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_block_data[tid] = fmaxf(s_block_data[tid], s_block_data[tid + s]);
            }
            __syncthreads();
        }
    }
    
    if (tid == 0) {
        d_scale[block_id] = s_block_data[0];
    }
    __syncthreads();

    float scale = d_scale[block_id];
    if (scale == 0.0f) scale = 1.0f; 

    for (int i = tid; i < block_size; i += blockDim.x) {
        float normalized_val = input[block_id * block_size + i] / scale;
        
        float min_dist = 1e9;
        uint8_t best_idx = 0;
        
        for (uint8_t j = 0; j < 15; ++j) { 
            float dist = fabsf(normalized_val - nf4_quantiles[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }
        
        uint32_t shift = (i % 2) * 4;
        uint32_t val_to_write = best_idx << shift;
        atomicOr((unsigned int*)(output + (i / 2)), val_to_write);
    }
}



__global__ void dequantize_from_nf4_kernel(const uint8_t* input, float* output, const float* d_scale, int block_size) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    float scale = d_scale[block_id];

    for (int i = tid; i < block_size; i += blockDim.x) {
        uint8_t packed_val = input[i / 2];
        uint8_t index = (i % 2 == 0) ? (packed_val & 0x0F) : (packed_val >> 4);
        
        float dequantized_val = nf4_quantiles[index];
        
        output[block_id * block_size + i] = dequantized_val * scale;
    }
}



void print_matrix(const std::string& name, const float* m, int rows, int cols) {
    std::cout << "--- " << name << " --- (" << rows << "x" << cols << ")\n";
    std::vector<float> h_m(rows * cols);
    CHECK_CUDA(cudaMemcpy(h_m.data(), m, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < std::min(4, rows); ++i) {
        for (int j = 0; j < std::min(8, cols); ++j) {
            std::cout << h_m[i * cols + j] << "\t";
        }
        std::cout << (cols > 8 ? "...\n" : "\n");
    }
    std::cout << (rows > 8 ? "...\n" : "");
    std::cout << "-----------------------\n" << std::endl;
}

float calculate_error(const float* ref, const float* quantized, int n_elements) {
    std::vector<float> h_ref(n_elements);
    std::vector<float> h_quantized(n_elements);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), ref, n_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_quantized.data(), quantized, n_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    double total_error = 0.0;
    for(int i=0; i<n_elements; ++i) {
        total_error += std::fabs(h_ref[i] - h_quantized[i]);
    }
    return static_cast<float>(total_error / n_elements);
}


int main() {
    const int total_weights = 1024 * 1024; 
    const int block_size = 64;             
    const int num_blocks = total_weights / block_size;

    std::vector<float> h_weights(total_weights);
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f); 
    for(float& v : h_weights) v = dis(gen);

    float *d_weights, *d_dequantized_weights, *d_scales;
    uint8_t* d_quantized_weights;
    
    CHECK_CUDA(cudaMalloc(&d_weights, total_weights * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dequantized_weights, total_weights * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scales, num_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_quantized_weights, total_weights / 2 * sizeof(uint8_t)));
    
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), total_weights * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "--- Original FP32 Weights ---" << std::endl;
    print_matrix("Original FP32 Weights", d_weights, 1, total_weights);
    
    dim3 quant_blocks(num_blocks);
    dim3 quant_threads(block_size);
    size_t shared_mem_size = block_size * sizeof(float); 
    CHECK_CUDA(cudaMemset(d_quantized_weights, 0, total_weights / 2 * sizeof(uint8_t))); // Important to zero out for atomicOr
    quantize_to_nf4_kernel<<<quant_blocks, quant_threads, shared_mem_size>>>(d_weights, d_quantized_weights, d_scales, block_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "--- 1. Quantization to NF4 Complete ---" << std::endl;

    dequantize_from_nf4_kernel<<<quant_blocks, quant_threads>>>(d_quantized_weights, d_dequantized_weights, d_scales, block_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "--- 2. Dequantization from NF4 Complete ---" << std::endl;
    print_matrix("Dequantized FP32 Weights", d_dequantized_weights, 1, total_weights);

    float avg_error = calculate_error(d_weights, d_dequantized_weights, total_weights);
    std::cout << "--- 3. Comparison ---" << std::endl;
    std::cout << "Average Absolute Error (FP32 vs. NF4): " << avg_error << std::endl;

    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_dequantized_weights));
    CHECK_CUDA(cudaFree(d_scales));
    CHECK_CUDA(cudaFree(d_quantized_weights));

    return 0;
}
