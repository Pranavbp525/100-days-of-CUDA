#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    const cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "cuDNN status: %s\n", cudnnGetErrorString(status)); \
        exit(1); \
    } \
}


__constant__ float nf4_quantiles[16] = {
    -1.0000f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f,
    0.0911f, 0.1848f, 0.2844f, 0.3949f, 0.5251f, 0.6962f, 1.0000f, 0.0f
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
            if (tid < s) s_block_data[tid] = fmaxf(s_block_data[tid], s_block_data[tid + s]);
            __syncthreads();
        }
    }
    if (tid == 0) d_scale[block_id] = s_block_data[0];
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


__global__ void qlora_gemm_forward_kernel(const float* input, const uint8_t* W_q, const float* scales, float* output, int M, int N, int K, int block_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            // Dequantize weight W[i, col] on the fly
            int weight_idx = i * N + col;
            int block_id = weight_idx / block_size;
            int idx_in_block = weight_idx % block_size;
            
            float scale = scales[block_id];
            uint8_t packed_val = W_q[weight_idx / 2];
            uint8_t quant_idx = (weight_idx % 2 == 0) ? (packed_val & 0x0F) : (packed_val >> 4);
            float w_dequant = nf4_quantiles[quant_idx] * scale;
            
            sum += input[row * K + i] * w_dequant;
        }
        output[row * N + col] = sum;
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


int main() {
    const int M = 128; 
    const int K = 4096; 
    const int N = 4096; 
    const int R = 16;  

    const int block_size = 64;
    const int num_blocks = (K * N) / block_size;

    std::vector<float> h_input(M * K);
    std::vector<float> h_W_base(K * N); 
    std::vector<float> h_lora_A(K * R);  
    std::vector<float> h_lora_B(R * N);  
    std::vector<float> h_grad_output(M * N);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for(float& v : h_input) v = dis(gen);
    for(float& v : h_W_base) v = dis(gen);
    for(float& v : h_lora_A) v = dis(gen);
    for(float& v : h_lora_B) v = 0.0f; 
    for(float& v : h_grad_output) v = dis(gen) * 0.01f;

    float *d_W_base_fp32, *d_scales;
    uint8_t* d_W_base_q;
    float *d_lora_A, *d_lora_B;
    float *d_input, *d_output, *d_base_output, *d_lora_output, *d_lora_intermediate;
    float *d_grad_output, *d_grad_lora_A, *d_grad_lora_B, *d_grad_lora_intermediate;

    CHECK_CUDA(cudaMalloc(&d_W_base_fp32, h_W_base.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scales, num_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_base_q, h_W_base.size() / 2 * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_lora_A, h_lora_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lora_B, h_lora_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_base_output, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lora_output, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lora_intermediate, M * R * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_lora_A, h_lora_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_lora_B, h_lora_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_lora_intermediate, M * R * sizeof(float)));
    
    // --- Copy & Quantize ---
    CHECK_CUDA(cudaMemcpy(d_W_base_fp32, h_W_base.data(), h_W_base.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_W_base_q, 0, h_W_base.size() / 2 * sizeof(uint8_t)));
    quantize_to_nf4_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_W_base_fp32, d_W_base_q, d_scales, block_size);
    CHECK_CUDA(cudaFree(d_W_base_fp32)); // Free the huge FP32 weights! This is the key memory saving.
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lora_A, h_lora_A.data(), h_lora_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lora_B, h_lora_B.data(), h_lora_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "--- 1. QLoRA Forward Pass ---" << std::endl;
    dim3 grid( (N + 15) / 16, (M + 15) / 16 );
    dim3 block(16, 16);
    qlora_gemm_forward_kernel<<<grid, block>>>(d_input, d_W_base_q, d_scales, d_base_output, M, N, K, block_size);
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    const float alpha = 1.0f, beta = 0.0f, beta_one = 1.0f;
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, R, M, K, &alpha, d_lora_A, R, d_input, K, &beta, d_lora_intermediate, R));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, R, &alpha, d_lora_B, N, d_lora_intermediate, R, &beta, d_lora_output, N));
    
    CHECK_CUBLAS(cublasSaxpy(cublas_handle, M * N, &alpha, d_lora_output, 1, d_base_output, 1));
    CHECK_CUDA(cudaMemcpy(d_output, d_base_output, M * N * sizeof(float), cudaMemcpyDeviceToDevice));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    print_matrix("QLoRA Forward Output", d_output, M, N);
    
    std::cout << "\n--- 2. QLoRA Backward Pass (Gradients for LoRA only) ---" << std::endl;
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, R, M, N, &alpha, d_lora_B, N, d_grad_output, N, &beta, d_grad_lora_intermediate, R));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, R, M, &alpha, d_grad_output, N, d_lora_intermediate, R, &beta, d_grad_lora_B, N));
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, K, R, M, &alpha, d_input, K, d_grad_lora_intermediate, R, &beta, d_grad_lora_A, K));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    print_matrix("Gradient w.r.t. LoRA A", d_grad_lora_A, K, R);
    print_matrix("Gradient w.r.t. LoRA B", d_grad_lora_B, R, N);

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaFree(d_scales)); CHECK_CUDA(cudaFree(d_W_base_q));
    CHECK_CUDA(cudaFree(d_lora_A)); CHECK_CUDA(cudaFree(d_lora_B));
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_base_output)); CHECK_CUDA(cudaFree(d_lora_output));
    CHECK_CUDA(cudaFree(d_lora_intermediate)); CHECK_CUDA(cudaFree(d_grad_output));
    CHECK_CUDA(cudaFree(d_grad_lora_A)); CHECK_CUDA(cudaFree(d_grad_lora_B));
    CHECK_CUDA(cudaFree(d_grad_lora_intermediate));

    return 0;
}
