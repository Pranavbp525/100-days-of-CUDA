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


// fused attention kernels

__global__ void fused_attention_forward_kernel(
    const float* q, const float* k, const float* v, float* output, float* lse, // lse is log-sum-exp for backward
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

    float acc[TILE_SIZE] = {0.0f};
    float max_score = -INFINITY;
    float sum_exp_scores = 0.0f;

    for (int key_block_start = 0; key_block_start < L; key_block_start += TILE_SIZE) {
        for (int i = threadIdx.x; i < TILE_SIZE * Dh; i += blockDim.x) {
            int key_row = key_block_start + i / Dh;
            int key_col = i % Dh;
            if (key_row < L) {
                int k_idx = (batch_idx * H * L + head_idx * L + key_row) * Dh + key_col;
                k_tile[i] = k[k_idx];
                v_tile[i] = v[k_idx];
            } else {
                k_tile[i] = 0.0f; v_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        float scores[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; ++j) {
            float score = 0.0f;
            for (int i = 0; i < Dh; ++i) score += q_vec[i] * k_tile[j * Dh + i];
            if (is_causal && (key_block_start + j > query_row_idx)) score = -1e9f;
            scores[j] = score * scale_factor;
        }

        float current_max = -INFINITY;
        for (int j = 0; j < TILE_SIZE; ++j) current_max = fmaxf(current_max, scores[j]);

        float old_max = max_score;
        max_score = fmaxf(max_score, current_max);
        
        if (old_max > -INFINITY) {
            float rescale = expf(old_max - max_score);
            sum_exp_scores *= rescale;
            for (int i = 0; i < Dh; ++i) acc[i] *= rescale;
        }

        for (int j = 0; j < TILE_SIZE; ++j) {
            float p_ij = expf(scores[j] - max_score);
            sum_exp_scores += p_ij;
            for (int i = 0; i < Dh; ++i) acc[i] += p_ij * v_tile[j * Dh + i];
        }
        __syncthreads();
    }

    lse[(batch_idx * H + head_idx) * L + query_row_idx] = max_score + logf(sum_exp_scores);

    for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
        int out_idx = (batch_idx * H * L + head_idx * L + query_row_idx) * Dh + i;
        output[out_idx] = acc[i] / sum_exp_scores;
    }
}


__global__ void fused_attention_backward_kernel(
    const float* q, const float* k, const float* v, const float* output, const float* lse, const float* d_output,
    float* d_q, float* d_k, float* d_v,
    int N, int H, int L, int Dh, float scale_factor, bool is_causal) {

    int batch_idx = blockIdx.x / L;
    int head_idx = blockIdx.y;
    int key_row_idx = blockIdx.x % L;

    const int TILE_SIZE = 128;
    extern __shared__ float shared_mem[];
    float* q_tile = shared_mem;
    float* d_q_tile = shared_mem + TILE_SIZE * Dh;

    float k_vec[TILE_SIZE];
    float v_vec[TILE_SIZE];
    for(int i = threadIdx.x; i < Dh; i+= blockDim.x) {
        int k_idx = (batch_idx * H * L + head_idx * L + key_row_idx) * Dh + i;
        k_vec[i] = k[k_idx];
        v_vec[i] = v[k_idx];
    }

    float d_k_vec[TILE_SIZE] = {0.0f};
    float d_v_vec[TILE_SIZE] = {0.0f};

    for (int query_block_start = 0; query_block_start < L; query_block_start += TILE_SIZE) {
        for (int i = threadIdx.x; i < TILE_SIZE * Dh; i += blockDim.x) {
            int query_row = query_block_start + i / Dh;
            int query_col = i % Dh;
            if (query_row < L) {
                int q_idx = (batch_idx * H * L + head_idx * L + query_row) * Dh + query_col;
                q_tile[i] = q[q_idx];
            } else {
                q_tile[i] = 0.0f;
            }
        }
        __syncthreads();
        
        float scores[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; j++) {
            float score = 0.0f;
            for (int i=0; i < Dh; i++) score += q_tile[j * Dh + i] * k_vec[i];
            if (is_causal && key_row_idx > query_block_start + j) score = -1e9f;
            scores[j] = score * scale_factor;
        }

        float p[TILE_SIZE];
        float lse_i[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; j++) {
            int lse_idx = (batch_idx * H + head_idx) * L + query_block_start + j;
            lse_i[j] = lse[lse_idx];
            p[j] = expf(scores[j] - lse_i[j]);
        }

        float d_p[TILE_SIZE] = {0.0f};
        for(int j=0; j < TILE_SIZE; j++) {
            for(int i=0; i < Dh; i++) {
                int d_out_idx = (batch_idx * H * L + head_idx * L + query_block_start + j) * Dh + i;
                int v_idx = (batch_idx * H * L + head_idx * L + key_row_idx) * Dh + i;
                d_p[j] += d_output[d_out_idx] * v[v_idx];
            }
        }

        float d_s[TILE_SIZE];
        for(int j=0; j < TILE_SIZE; j++){
            float o_vec[TILE_SIZE];
            for(int i=0; i < Dh; i++) {
                int o_idx = (batch_idx * H * L + head_idx * L + query_block_start + j) * Dh + i;
                o_vec[i] = output[o_idx];
            }
            float do_dv = 0;
            for(int i=0; i < Dh; i++) {
                 int d_out_idx = (batch_idx * H * L + head_idx * L + query_block_start + j) * Dh + i;
                 do_dv += d_output[d_out_idx] * o_vec[i];
            }
            d_s[j] = scale_factor * p[j] * (d_p[j] - do_dv);
        }

        for (int j = 0; j < TILE_SIZE; j++) {
            for (int i = 0; i < Dh; i++) {
                 d_q_tile[j * Dh + i] += d_s[j] * k_vec[i];
                 d_k_vec[i] += d_s[j] * q_tile[j * Dh + i];
            }
        }
        
        for(int j = 0; j < TILE_SIZE; j++) {
            for(int i=0; i < Dh; i++) {
                int d_out_idx = (batch_idx * H * L + head_idx * L + query_block_start + j) * Dh + i;
                d_v_vec[i] += p[j] * d_output[d_out_idx];
            }
        }
        __syncthreads();
    }

    for(int i = threadIdx.x; i < Dh; i += blockDim.x){
        int d_k_idx = (batch_idx * H * L + head_idx * L + key_row_idx) * Dh + i;
        atomicAdd(&d_k[d_k_idx], d_k_vec[i]);
        atomicAdd(&d_v[d_k_idx], d_v_vec[i]);
    }
}

// utility kernels
__global__ void concat_heads_kernel(const float* input, float* output, int N, int H, int L, int Dh) {
    int n = blockIdx.x / L; int l = blockIdx.x % L;
    for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
            int input_idx = n * H * L * Dh + h * L * Dh + l * Dh + d;
            int output_idx = n * L * (H * Dh) + l * (H * Dh) + h * Dh + d;
            output[output_idx] = input[input_idx];
        }
    }
}
__global__ void split_heads_grad_kernel(const float* grad_in, float* grad_out, int N, int H, int L, int Dh) {
    int n = blockIdx.x / L; int l = blockIdx.x % L;
    for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
            int input_idx = n * L * (H * Dh) + l * (H * Dh) + h * Dh + d;
            int output_idx = n * H * L * Dh + h * L * Dh + l * Dh + d;
            grad_out[output_idx] = grad_in[input_idx];
        }
    }
}
__global__ void add_matrices_kernel(float* out, const float* in1, const float* in2, const float* in3, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) out[i] = in1[i] + in2[i] + in3[i];
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
    const int batch_size = 4, seq_len = 256, embed_dim = 512, num_heads = 8;
    const int head_dim = embed_dim / num_heads;

    std::vector<float> h_input(batch_size*seq_len*embed_dim), h_Wq(embed_dim*embed_dim), h_Wk(embed_dim*embed_dim), h_Wv(embed_dim*embed_dim), h_Wo(embed_dim*embed_dim), h_grad_output(h_input.size());

    float *d_input, *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_Q, *d_K, *d_V, *d_lse;
    float *d_attn_heads_output, *d_concat_heads, *d_output;
    float *d_grad_output, *d_grad_concat_heads, *d_grad_Wo, *d_grad_attn_heads_output;
    float *d_grad_V, *d_grad_Q, *d_grad_K, *d_grad_Wq, *d_grad_Wk, *d_grad_Wv, *d_grad_input;
    float *d_grad_input_from_Q, *d_grad_input_from_K, *d_grad_input_from_V;
    
    CHECK_CUDA(cudaMalloc(&d_lse, batch_size * num_heads * seq_len * sizeof(float)));


    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    const float alpha = 1.0f, beta = 0.0f;
    const int N_in = batch_size * seq_len;

    // forward pass
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V, embed_dim));
    
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const bool is_causal = true;
    dim3 grid(batch_size * seq_len, num_heads, 1);
    dim3 block(128, 1, 1);
    size_t shared_mem_size = 2 * 128 * head_dim * sizeof(float);
    fused_attention_forward_kernel<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_attn_heads_output, d_lse, batch_size, num_heads, seq_len, head_dim, scale_factor, is_causal);
    
    dim3 concat_blocks(N_in);
    dim3 concat_threads(32, 8);
    concat_heads_kernel<<<concat_blocks, concat_threads>>>(d_attn_heads_output, d_concat_heads, batch_size, num_heads, seq_len, head_dim);
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_concat_heads, embed_dim, &beta, d_output, embed_dim));
    
    // backward pass
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, embed_dim, N_in, &alpha, d_grad_output, embed_dim, d_concat_heads, embed_dim, &beta, d_grad_Wo, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_grad_output, embed_dim, &beta, d_grad_concat_heads, embed_dim));
    split_heads_grad_kernel<<<concat_blocks, concat_threads>>>(d_grad_concat_heads, d_grad_attn_heads_output, batch_size, num_heads, seq_len, head_dim);

    CHECK_CUDA(cudaMemset(d_grad_Q, 0, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_K, 0, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_V, 0, h_input.size() * sizeof(float)));
    dim3 bwd_grid(batch_size * seq_len, num_heads, 1);
    dim3 bwd_block(128, 1, 1);
    size_t bwd_shared_mem_size = 2 * 128 * head_dim * sizeof(float);
    fused_attention_backward_kernel<<<bwd_grid, bwd_block, bwd_shared_mem_size>>>(
        d_Q, d_K, d_V, d_output, d_lse, d_grad_attn_heads_output,
        d_grad_Q, d_grad_K, d_grad_V,
        batch_size, num_heads, seq_len, head_dim, scale_factor, is_causal
    );

    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, N_in, &alpha, d_grad_Q, embed_dim, d_input, embed_dim, &beta, d_grad_Wq, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, N_in, &alpha, d_grad_K, embed_dim, d_input, embed_dim, &beta, d_grad_Wk, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, N_in, &alpha, d_grad_V, embed_dim, d_input, embed_dim, &beta, d_grad_Wv, embed_dim));
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_grad_Q, embed_dim, &beta, d_grad_input_from_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_grad_K, embed_dim, &beta, d_grad_input_from_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_grad_V, embed_dim, &beta, d_grad_input_from_V, embed_dim));
    add_matrices_kernel<<<(h_input.size() + 255) / 256, 256>>>(d_grad_input, d_grad_input_from_Q, d_grad_input_from_K, d_grad_input_from_V, h_input.size());

    std::cout << "Backward pass complete." << std::endl;
    print_matrix("Gradient w.r.t. Input", d_grad_input, N_in, embed_dim);


    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
