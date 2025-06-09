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



__global__ void flash_attention_forward_kernel(
    const float* q_in, const float* k_in, const float* v_in, float* output,
    float* lse, 
    int N, int H, int L, int Dh, float scale_factor, bool is_causal) {

    
    const int TILE_SIZE_Q = 64;
    const int TILE_SIZE_K = 64;

    
    int query_row_idx = blockIdx.x;
    int batch_idx = query_row_idx / L;
    int head_idx = blockIdx.y;
    int local_query_row = query_row_idx % L;


    extern __shared__ float shared_mem[];
    float* k_tile = shared_mem;
    float* v_tile = shared_mem + TILE_SIZE_K * Dh;
    
    
    float acc[TILE_SIZE_K]; 
    float max_score = -INFINITY;
    float sum_exp_scores = 0.0f;
    for(int i=0; i < Dh; ++i) {
        acc[i] = 0.0f;
    }
    
    float q_vec[TILE_SIZE_K];
    int q_offset = (batch_idx * H * L + head_idx * L + local_query_row) * Dh;
    for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
        q_vec[i] = q_in[q_offset + i];
    }
    __syncthreads(); 


    for (int block_k_start = 0; block_k_start < L; block_k_start += TILE_SIZE_K) {
        for (int i = threadIdx.x; i < TILE_SIZE_K * Dh; i += blockDim.x) {
            int row = i / Dh;
            int col = i % Dh;
            int k_row_idx = block_k_start + row;
            if (k_row_idx < L) {
                int k_idx = (batch_idx * H * L + head_idx * L + k_row_idx) * Dh + col;
                k_tile[row * Dh + col] = k_in[k_idx];
                v_tile[row * Dh + col] = v_in[k_idx];
            } else {
                 k_tile[row * Dh + col] = 0.0;
                 v_tile[row * Dh + col] = 0.0;
            }
        }
        __syncthreads();

   
        float scores[TILE_SIZE_K];
        for (int k_idx_in_tile = 0; k_idx_in_tile < TILE_SIZE_K; ++k_idx_in_tile) {
            float score = 0.0f;
            for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
                 score += q_vec[i] * k_tile[k_idx_in_tile * Dh + i];
            }
             __syncthreads(); 
            
            
            score = 0;
            for(int i=0; i<Dh; ++i) score += q_vec[i] * k_tile[k_idx_in_tile * Dh + i];


            score *= scale_factor;

            if (is_causal && (block_k_start + k_idx_in_tile > local_query_row)) {
                score = -INFINITY;
            }
            scores[k_idx_in_tile] = score;
        }

        float old_max = max_score;
        for(int i=0; i<TILE_SIZE_K; ++i) max_score = fmaxf(max_score, scores[i]);
        
        float scale = expf(old_max - max_score);
        sum_exp_scores *= scale;
        for(int i=0; i<Dh; ++i) acc[i] *= scale;

        for (int k_idx_in_tile = 0; k_idx_in_tile < TILE_SIZE_K; ++k_idx_in_tile) {
            float p = expf(scores[k_idx_in_tile] - max_score);
            sum_exp_scores += p;
            for(int i=0; i<Dh; ++i) {
                acc[i] += p * v_tile[k_idx_in_tile * Dh + i];
            }
        }
        __syncthreads();
    }
    
    float inv_sum = 1.0f / sum_exp_scores;
    for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
        int out_idx = (batch_idx * H * L + head_idx * L + local_query_row) * Dh + i;
        output[out_idx] = acc[i] * inv_sum;
    }

    if (threadIdx.x == 0) {
        lse[blockIdx.y * L + query_row_idx] = max_score + logf(sum_exp_scores);
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

__global__ void final_projection_kernel(const float* input, float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size) output[i] = input[i]; 
}


int main() {
    const int batch_size = 4;
    const int seq_len = 1024; 
    const int embed_dim = 512;
    const int num_heads = 8;
    
    if (embed_dim % num_heads != 0) {
        std::cerr << "Embedding dimension must be divisible by the number of heads." << std::endl;
        return 1;
    }
    const int head_dim = embed_dim / num_heads;

    std::cout << "--- FlashAttention-Style Forward Pass ---" << std::endl;

    std::vector<float> h_input(batch_size * seq_len * embed_dim);
    std::vector<float> h_Wq(embed_dim * embed_dim), h_Wk(embed_dim * embed_dim), h_Wv(embed_dim * embed_dim), h_Wo(embed_dim * embed_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& v : h_input) v = dis(gen);
    for (float& v : h_Wq) v = dis(gen);
    for (float& v : h_Wk) v = dis(gen);
    for (float& v : h_Wv) v = dis(gen);
    for (float& v : h_Wo) v = dis(gen);

    float *d_input, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V;
    float *d_output, *d_lse, *d_attn_heads_output;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq, h_Wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, h_Wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, h_Wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, h_Wo.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_heads_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lse, batch_size * num_heads * seq_len * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo.data(), h_Wo.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f;
    const int N_in = batch_size * seq_len;

    std::cout << "\n--- 1. Forward Pass ---\n" << std::endl;
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V, embed_dim));
    
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const bool is_causal = true;
    
    dim3 grid(batch_size * seq_len, num_heads, 1);
    dim3 block(128); 
    size_t shared_mem_size = 2 * 128 * head_dim * sizeof(float);
    
    std::cout << "Launching FlashAttention-style kernel..." << std::endl;
    flash_attention_forward_kernel<<<grid, block, shared_mem_size>>>(
        d_Q, d_K, d_V, d_attn_heads_output, d_lse,
        batch_size, num_heads, seq_len, head_dim,
        scale_factor, is_causal
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    
    final_projection_kernel<<<(N_in*embed_dim+255)/256, 256>>>(d_attn_heads_output, d_output, N_in*embed_dim);
    CHECK_CUDA(cudaDeviceSynchronize());


    std::cout << "FlashAttention-style forward pass kernel complete." << std::endl << std::endl;
    print_matrix("Final Output (from Fused Kernel)", d_output, N_in, embed_dim);

    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_Wq)); CHECK_CUDA(cudaFree(d_Wk)); CHECK_CUDA(cudaFree(d_Wv)); CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_K)); CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_output)); CHECK_CUDA(cudaFree(d_lse)); CHECK_CUDA(cudaFree(d_attn_heads_output));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
