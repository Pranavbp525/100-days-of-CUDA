#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
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


__global__ void softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x;
    extern __shared__ float sdata[];

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, data[row * cols + i]);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += expf(data[row * cols + i] - max_val);
    }
    sdata[threadIdx.x] = sum;
     __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        data[row * cols + i] = expf(data[row * cols + i] - max_val) / sum;
    }
}

__global__ void top1_routing_kernel(const float* probs, int* expert_indices, float* scores, int num_tokens, int num_experts) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < num_tokens) {
        float max_prob = -1.0f;
        int top_expert = 0;
        for (int i = 0; i < num_experts; ++i) {
            if (probs[token_idx * num_experts + i] > max_prob) {
                max_prob = probs[token_idx * num_experts + i];
                top_expert = i;
            }
        }
        expert_indices[token_idx] = top_expert;
        scores[token_idx] = max_prob;
    }
}

__global__ void gather_tokens_kernel(const float* input, float* expert_input, const int* expert_indices, const int* dispatch_indices, int num_tokens_for_expert, int embed_dim) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < num_tokens_for_expert) {
        int original_token_idx = dispatch_indices[token_idx];
        for (int i = 0; i < embed_dim; ++i) {
            expert_input[token_idx * embed_dim + i] = input[original_token_idx * embed_dim + i];
        }
    }
}

__global__ void scatter_tokens_kernel(const float* expert_output, float* final_output, const int* dispatch_indices, const float* scores, int num_tokens_for_expert, int embed_dim) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < num_tokens_for_expert) {
        int original_token_idx = dispatch_indices[token_idx];
        float score = scores[original_token_idx];
        for (int i = 0; i < embed_dim; ++i) {
            final_output[original_token_idx * embed_dim + i] = expert_output[token_idx * embed_dim + i] * score;
        }
    }
}

__global__ void scatter_add_grad_kernel(const float* expert_grad_input, float* final_grad_input, const int* dispatch_indices, const float* scores, int num_tokens_for_expert, int embed_dim) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < num_tokens_for_expert) {
        int original_token_idx = dispatch_indices[token_idx];
        float score = scores[original_token_idx];
        for (int i = 0; i < embed_dim; ++i) {
            atomicAdd(&final_grad_input[original_token_idx * embed_dim + i], expert_grad_input[token_idx * embed_dim + i] * score);
        }
    }
}

void print_matrix(const std::string& name, const float* m, int rows, int cols) {
    std::cout << "--- " << name << " --- (" << rows << "x" << cols << ")\n";
    std::vector<float> h_m(rows * cols);
    CHECK_CUDA(cudaMemcpy(h_m.data(), m, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < std::min(4, rows); ++i) {
        for (int j = 0; j < std::min(8, cols); ++j) std::cout << h_m[i * cols + j] << "\t";
        std::cout << (cols > 8 ? "...\n" : "\n");
    }
    std::cout << (rows > 8 ? "...\n" : "");
    std::cout << "-----------------------\n" << std::endl;
}

int main() {
    const int batch_size = 4, seq_len = 64, embed_dim = 512;
    const int num_experts = 8, num_tokens = batch_size * seq_len;

    std::vector<float> h_input(num_tokens * embed_dim), h_W_gate(embed_dim * num_experts), h_grad_output(num_tokens * embed_dim);
    std::vector<std::vector<float>> h_expert_weights(num_experts, std::vector<float>(embed_dim * embed_dim));
    std::mt19937 gen(42); std::normal_distribution<float> dis(0.0f, 0.02f);
    for(auto& v : h_input) v = dis(gen); for(auto& v : h_W_gate) v = dis(gen);
    for(auto& expert_w : h_expert_weights) for(auto& v : expert_w) v = dis(gen);
    for(auto& v : h_grad_output) v = dis(gen);

    float *d_input, *d_W_gate, *d_logits, *d_final_output, *d_grad_output, *d_grad_W_gate, *d_grad_input;
    int* d_expert_indices; float* d_scores;
    std::vector<float*> d_expert_weights(num_experts);
    std::vector<float*> d_grad_expert_weights(num_experts);

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_gate, h_W_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_final_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_W_gate, h_W_gate.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_indices, num_tokens * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_scores, num_tokens * sizeof(float)));
    for(int i=0; i<num_experts; ++i) {
        CHECK_CUDA(cudaMalloc(&d_expert_weights[i], embed_dim * embed_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grad_expert_weights[i], embed_dim * embed_dim * sizeof(float)));
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_gate, h_W_gate.data(), h_W_gate.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice));
    for(int i=0; i<num_experts; ++i) CHECK_CUDA(cudaMemcpy(d_expert_weights[i], h_expert_weights[i].data(), embed_dim * embed_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_final_output, 0, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_input, 0, h_input.size() * sizeof(float)));
    
    cublasHandle_t cublas_handle; CHECK_CUBLAS(cublasCreate(&cublas_handle));
    const float alpha = 1.0f, beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, num_experts, num_tokens, embed_dim, &alpha, d_W_gate, embed_dim, d_input, embed_dim, &beta, d_logits, num_experts));
    softmax_kernel<<<num_tokens, 256, 256*sizeof(float)>>>(d_logits, num_tokens, num_experts);
    
    top1_routing_kernel<<<(num_tokens + 255) / 256, 256>>>(d_logits, d_expert_indices, d_scores, num_tokens, num_experts);
    
    std::vector<int> h_expert_indices(num_tokens);
    CHECK_CUDA(cudaMemcpy(h_expert_indices.data(), d_expert_indices, num_tokens * sizeof(int), cudaMemcpyDeviceToHost));
    
    for(int i=0; i<num_experts; ++i) {
        std::vector<int> dispatch_indices_h;
        for(int j=0; j<num_tokens; ++j) {
            if (h_expert_indices[j] == i) dispatch_indices_h.push_back(j);
        }
        if (dispatch_indices_h.empty()) continue;

        int num_tokens_for_expert = dispatch_indices_h.size();
        int* d_dispatch_indices;
        float *d_expert_input, *d_expert_output;
        CHECK_CUDA(cudaMalloc(&d_dispatch_indices, num_tokens_for_expert * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_expert_input, num_tokens_for_expert * embed_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_expert_output, num_tokens_for_expert * embed_dim * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_dispatch_indices, dispatch_indices_h.data(), num_tokens_for_expert * sizeof(int), cudaMemcpyHostToDevice));

        gather_tokens_kernel<<<(num_tokens_for_expert + 255) / 256, 256>>>(d_input, d_expert_input, d_expert_indices, d_dispatch_indices, num_tokens_for_expert, embed_dim);
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, num_tokens_for_expert, embed_dim, &alpha, d_expert_weights[i], embed_dim, d_expert_input, embed_dim, &beta, d_expert_output, embed_dim));
        scatter_tokens_kernel<<<(num_tokens_for_expert + 255) / 256, 256>>>(d_expert_output, d_final_output, d_dispatch_indices, d_scores, num_tokens_for_expert, embed_dim);
        
        CHECK_CUDA(cudaFree(d_dispatch_indices)); CHECK_CUDA(cudaFree(d_expert_input)); CHECK_CUDA(cudaFree(d_expert_output));
    }
    
    for(int i=0; i<num_experts; ++i) {
        std::vector<int> dispatch_indices_h;
        for(int j=0; j<num_tokens; ++j) if (h_expert_indices[j] == i) dispatch_indices_h.push_back(j);
        if (dispatch_indices_h.empty()) continue;

        int num_tokens_for_expert = dispatch_indices_h.size();
        int* d_dispatch_indices;
        float *d_expert_input, *d_grad_expert_output, *d_grad_expert_input;
        CHECK_CUDA(cudaMalloc(&d_dispatch_indices, num_tokens_for_expert * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_expert_input, num_tokens_for_expert * embed_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grad_expert_output, num_tokens_for_expert * embed_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grad_expert_input, num_tokens_for_expert * embed_dim * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_dispatch_indices, dispatch_indices_h.data(), num_tokens_for_expert * sizeof(int), cudaMemcpyHostToDevice));
        
        gather_tokens_kernel<<<(num_tokens_for_expert + 255) / 256, 256>>>(d_grad_output, d_grad_expert_output, d_expert_indices, d_dispatch_indices, num_tokens_for_expert, embed_dim);
        gather_tokens_kernel<<<(num_tokens_for_expert + 255) / 256, 256>>>(d_input, d_expert_input, d_expert_indices, d_dispatch_indices, num_tokens_for_expert, embed_dim);

        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, num_tokens_for_expert, &alpha, d_grad_expert_output, embed_dim, d_expert_input, embed_dim, &beta, d_grad_expert_weights[i], embed_dim));
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, num_tokens_for_expert, embed_dim, &alpha, d_expert_weights[i], embed_dim, d_grad_expert_output, embed_dim, &beta, d_grad_expert_input, embed_dim));
        scatter_add_grad_kernel<<<(num_tokens_for_expert + 255) / 256, 256>>>(d_grad_expert_input, d_grad_input, d_dispatch_indices, d_scores, num_tokens_for_expert, embed_dim);

        CHECK_CUDA(cudaFree(d_dispatch_indices)); CHECK_CUDA(cudaFree(d_expert_input)); CHECK_CUDA(cudaFree(d_grad_expert_output)); CHECK_CUDA(cudaFree(d_grad_expert_input));
    }
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, num_experts, embed_dim, num_tokens, &alpha, d_logits, num_experts, d_input, embed_dim, &beta, d_grad_W_gate, num_experts));

    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
