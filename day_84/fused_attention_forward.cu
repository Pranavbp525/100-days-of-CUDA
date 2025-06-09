#include <iostream>
#include <vector>
#include <random>
#include <cmath>
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
        fprintf(stderr, "cublasStatus: %d\n", status); \
        exit(1); \
    } \
}


// fused attention kernel
// S = softmax(Q @ K^T / sqrt(d_k)) @ V
// shared memory tiling to reduce global memory reads.
// each thread block processes one row of the query matrix (one token's attention).
__global__ void fused_attention_forward_kernel(
    const float* q, const float* k, const float* v, float* output,
    int N, int H, int L, int Dh, float scale_factor, bool is_causal) {

    int batch_idx = blockIdx.x / L;
    int head_idx = blockIdx.y;
    int query_row_idx = blockIdx.x % L;

    
    const int TILE_SIZE = 128; 
    extern __shared__ float shared_mem[];
    float* k_tile = shared_mem;
    float* v_tile = shared_mem + TILE_SIZE * Dh;

    float q_vec[TILE_SIZE]; 
    for(int i = threadIdx.x; i < Dh; i += blockDim.x) {
        int q_idx = (batch_idx * H * L + head_idx * L + query_row_idx) * Dh + i;
        q_vec[i] = q[q_idx];
    }

    float scores[TILE_SIZE]; 
    float max_score = -INFINITY;
    float sum_exp_scores = 0.0f;
    float acc[TILE_SIZE] = {0.0f}; 

    // iterate over key/value pairs in tiles
    for (int key_block_start = 0; key_block_start < L; key_block_start += TILE_SIZE) {
        // load K and V tiles into shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * Dh; i += blockDim.x) {
            int key_row = key_block_start + i / Dh;
            int key_col = i % Dh;
            if (key_row < L) {
                int k_idx = (batch_idx * H * L + head_idx * L + key_row) * Dh + key_col;
                k_tile[i] = k[k_idx];
                v_tile[i] = v[k_idx];
            } else {
                k_tile[i] = 0.0f;
                v_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        // compute scores S = Q @ K^T for the tile
        for (int j = 0; j < TILE_SIZE; ++j) {
            float score = 0.0f;
            for (int i = 0; i < Dh; ++i) {
                score += q_vec[i] * k_tile[j * Dh + i];
            }
            
            
            if (is_causal && (key_block_start + j > query_row_idx)) {
                score = -1e9f;
            }
            scores[j] = score * scale_factor;
        }

        // Online softmax
        // Numerically stable softmax calculation within the loop
        float current_max = -INFINITY;
        for (int j = 0; j < TILE_SIZE; ++j) {
            current_max = fmaxf(current_max, scores[j]);
        }

        float old_max = max_score;
        max_score = fmaxf(max_score, current_max);
        
        // Rescale accumulator and sum based on new max
        if (old_max > -INFINITY) {
            float rescale = expf(old_max - max_score);
            sum_exp_scores *= rescale;
            for (int i = 0; i < Dh; ++i) {
                acc[i] *= rescale;
            }
        }

        float current_sum = 0.0f;
        for (int j = 0; j < TILE_SIZE; ++j) {
            float p_ij = expf(scores[j] - max_score);
            current_sum += p_ij;
            // Update accumulator: O += P_ij * V_j
            for (int i = 0; i < Dh; ++i) {
                acc[i] += p_ij * v_tile[j * Dh + i];
            }
        }
        sum_exp_scores += current_sum;
        __syncthreads(); 
    }

    // final normalization and write to output
    for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
        int out_idx = (batch_idx * H * L + head_idx * L + query_row_idx) * Dh + i;
        output[out_idx] = acc[i] / sum_exp_scores;
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
    std::cout << (rows > 4 ? "...\n" : "");
    std::cout << "-----------------------\n" << std::endl;
}


int main() {
    // multi-head attention dimensions
    const int batch_size = 4;
    const int seq_len = 256;
    const int embed_dim = 512;
    const int num_heads = 8;
    
    if (embed_dim % num_heads != 0) {
        std::cerr << "Embedding dimension must be divisible by the number of heads." << std::endl;
        return 1;
    }
    const int head_dim = embed_dim / num_heads;

    std::cout << "--- Fused Multi-Head Attention Forward Pass ---" << std::endl;

    // host data initialization
    std::vector<float> h_input(batch_size * seq_len * embed_dim);
    std::vector<float> h_Wq(embed_dim * embed_dim), h_Wk(embed_dim * embed_dim), h_Wv(embed_dim * embed_dim), h_Wo(embed_dim * embed_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& v : h_input) v = dis(gen);
    for (float& v : h_Wq) v = dis(gen);
    for (float& v : h_Wk) v = dis(gen);
    for (float& v : h_Wv) v = dis(gen);
    for (float& v : h_Wo) v = dis(gen);

        // --- Device Memory Allocation ---
    float *d_input, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V;
    float *d_attn_heads_output, *d_concat_heads, *d_output;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq, h_Wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, h_Wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, h_Wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, h_Wo.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_heads_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_concat_heads, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_input.size() * sizeof(float)));

    // copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo.data(), h_Wo.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f;
    const int N_in = batch_size * seq_len;

    // forward pass
    std::cout << "\n--- 1. Forward Pass ---\n" << std::endl;
    
    // project inputs to Q, K, V (using cuBLAS)
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V, embed_dim));
    
    // launch single fused attention kernel
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const bool is_causal = true; // Set to true for decoder-style masked attention
    
    dim3 grid(batch_size * seq_len, num_heads, 1);
    dim3 block(128, 1, 1);
    size_t shared_mem_size = 2 * 128 * head_dim * sizeof(float); // Tiled K and V
    
    fused_attention_forward_kernel<<<grid, block, shared_mem_size>>>(
        d_Q, d_K, d_V, d_attn_heads_output,
        batch_size, num_heads, seq_len, head_dim,
        scale_factor, is_causal
    );
    
    // concat heads

    dim3 concat_blocks(N_in);
    dim3 concat_threads(32, 8); // 256 threads
    concat_heads_kernel<<<concat_blocks, concat_threads>>>(d_attn_heads_output, d_concat_heads, batch_size, num_heads, seq_len, head_dim);
    // linear projection
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_concat_heads, embed_dim, &beta, d_output, embed_dim));

    std::cout << "Fused Attention forward pass complete." << std::endl << std::endl;
    print_matrix("Final Output", d_output, N_in, embed_dim);

    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_Wq)); CHECK_CUDA(cudaFree(d_Wk)); CHECK_CUDA(cudaFree(d_Wv)); CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_K)); CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_attn_heads_output)); CHECK_CUDA(cudaFree(d_concat_heads)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
