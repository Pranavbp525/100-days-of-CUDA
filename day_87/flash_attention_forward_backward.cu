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

    int query_row_idx_global = blockIdx.x;
    int batch_idx = query_row_idx_global / L;
    int head_idx = blockIdx.y;
    int local_query_row = query_row_idx_global % L;

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
        #pragma unroll
        for (int k_idx_in_tile = 0; k_idx_in_tile < TILE_SIZE_K; ++k_idx_in_tile) {
            float score = 0.0;
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
    
    int lse_idx = (batch_idx * H + head_idx) * L + local_query_row;
    lse[lse_idx] = max_score + logf(sum_exp_scores);

    float inv_sum = 1.0f / sum_exp_scores;
    for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
        int out_idx = (batch_idx * H * L + head_idx * L + local_query_row) * Dh + i;
        output[out_idx] = acc[i] * inv_sum;
    }
}

__global__ void flash_attention_backward_kernel(
    const float* q_in, const float* k_in, const float* v_in, const float* output, const float* lse, 
    const float* d_output,
    float* d_q, float* d_k, float* d_v,
    int N, int H, int L, int Dh, float scale_factor, bool is_causal) {
    
    const int TILE_SIZE_Q = 64;
    const int TILE_SIZE_K = 64;

    int query_row_idx_global = blockIdx.x;
    int batch_idx = query_row_idx_global / L;
    int head_idx = blockIdx.y;
    int local_query_row = query_row_idx_global % L;

    extern __shared__ float shared_mem[];
    float* k_tile = shared_mem;
    float* v_tile = shared_mem + TILE_SIZE_K * Dh;
    float* d_k_tile = shared_mem + 2 * TILE_SIZE_K * Dh;
    float* d_v_tile = shared_mem + 3 * TILE_SIZE_K * Dh;

    float q_vec[TILE_SIZE_K];
    int q_offset = (batch_idx * H * L + head_idx * L + local_query_row) * Dh;
    for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
        q_vec[i] = q_in[q_offset + i];
    }

    float d_q_acc[TILE_SIZE_K] = {0.0f};
    float d_o_vec[TILE_SIZE_K], o_vec[TILE_SIZE_K];
    int o_offset = (batch_idx * H * L + head_idx * L + local_query_row) * Dh;
    for (int i = threadIdx.x; i < Dh; i+= blockDim.x) {
        d_o_vec[i] = d_output[o_offset + i];
        o_vec[i] = output[o_offset + i];
    }
    
    float d_dot_o = 0.0f;
    for(int i=0; i < Dh; ++i) d_dot_o += d_o_vec[i] * o_vec[i];

    for (int block_k_start = 0; block_k_start < L; block_k_start += TILE_SIZE_K) {
        for (int i = threadIdx.x; i < TILE_SIZE_K * Dh; i += blockDim.x) {
            int row = i / Dh;
            int col = i % Dh;
            int k_row_idx = block_k_start + row;
            if (k_row_idx < L) {
                int k_idx = (batch_idx * H * L + head_idx * L + k_row_idx) * Dh + col;
                k_tile[row * Dh + col] = k_in[k_idx];
                v_tile[row * Dh + col] = v_in[k_idx];
                d_k_tile[row * Dh + col] = 0.0f;
                d_v_tile[row * Dh + col] = 0.0f;
            }
        }
        __syncthreads();

        float scores[TILE_SIZE_K];
        for (int k_idx_in_tile = 0; k_idx_in_tile < TILE_SIZE_K; ++k_idx_in_tile) {
            float score = 0.0;
            for(int i=0; i<Dh; ++i) score += q_vec[i] * k_tile[k_idx_in_tile * Dh + i];
            score *= scale_factor;
            if (is_causal && (block_k_start + k_idx_in_tile > local_query_row)) score = -INFINITY;
            scores[k_idx_in_tile] = score;
        }

        int lse_idx = (batch_idx * H + head_idx) * L + local_query_row;
        float lse_val = lse[lse_idx];
        
        for (int k_idx_in_tile = 0; k_idx_in_tile < TILE_SIZE_K; ++k_idx_in_tile) {
            float p_ij = expf(scores[k_idx_in_tile] - lse_val);
            float d_p_ij = 0.0f;
            for(int i=0; i<Dh; ++i) d_p_ij += d_o_vec[i] * v_tile[k_idx_in_tile * Dh + i];
            
            float d_s_ij = p_ij * (d_p_ij - d_dot_o) * scale_factor;

            for(int i=0; i<Dh; ++i) {
                d_q_acc[i] += d_s_ij * k_tile[k_idx_in_tile * Dh + i];
                d_k_tile[k_idx_in_tile * Dh + i] += d_s_ij * q_vec[i];
                d_v_tile[k_idx_in_tile * Dh + i] += p_ij * d_o_vec[i];
            }
        }

        for (int i = threadIdx.x; i < TILE_SIZE_K * Dh; i += blockDim.x) {
             int row = i / Dh;
             int k_row_idx = block_k_start + row;
             if (k_row_idx < L) {
                 int k_idx = (batch_idx * H * L + head_idx * L + k_row_idx) * Dh + (i % Dh);
                 atomicAdd(&d_k[k_idx], d_k_tile[i]);
                 atomicAdd(&d_v[k_idx], d_v_tile[i]);
             }
        }
        __syncthreads();
    }
    
    for (int i = threadIdx.x; i < Dh; i += blockDim.x) {
        atomicAdd(&d_q[q_offset + i], d_q_acc[i]);
    }
}


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
        for (int j = 0; j < std::min(8, cols); ++j) std::cout << h_m[i * cols + j] << "\t";
        std::cout << (cols > 8 ? "...\n" : "\n");
    }
    std::cout << (rows > 4 ? "...\n" : "");
    std::cout << "-----------------------\n" << std::endl;
}

int main() {
    const int batch_size = 4, seq_len = 1024, embed_dim = 512, num_heads = 8;
    const int head_dim = embed_dim / num_heads;

    std::vector<float> h_input(batch_size*seq_len*embed_dim), h_Wq(embed_dim*embed_dim), h_Wk(embed_dim*embed_dim), h_Wv(embed_dim*embed_dim), h_Wo(embed_dim*embed_dim), h_grad_output(h_input.size());
    std::mt19937 gen(42); std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& v : h_input) v = dis(gen); for (float& v : h_Wq) v = dis(gen); for (float& v : h_Wk) v = dis(gen);
    for (float& v : h_Wv) v = dis(gen); for (float& v : h_Wo) v = dis(gen); for (float& v : h_grad_output) v = dis(gen);

    float *d_input, *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_Q, *d_K, *d_V, *d_lse, *d_attn_heads_output, *d_concat_heads, *d_output;
    float *d_grad_output, *d_grad_concat_heads, *d_grad_Wo, *d_grad_attn_heads_output, *d_grad_V, *d_grad_Q, *d_grad_K;
    float *d_grad_Wq, *d_grad_Wk, *d_grad_Wv, *d_grad_input, *d_grad_input_from_Q, *d_grad_input_from_K, *d_grad_input_from_V;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_Wq, h_Wq.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, h_Wk.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_Wv, h_Wv.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, h_Wo.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_Q, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_V, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_heads_output, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_concat_heads, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_lse, batch_size*num_heads*seq_len*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_output, h_grad_output.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_grad_concat_heads, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_Wo, h_Wo.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_grad_attn_heads_output, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_V, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_grad_Q, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_K, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_grad_Wq, h_Wq.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_Wk, h_Wk.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_grad_Wv, h_Wv.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_grad_input_from_Q, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input_from_K, h_input.size()*sizeof(float))); CHECK_CUDA(cudaMalloc(&d_grad_input_from_V, h_input.size()*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo.data(), h_Wo.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size()*sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    const float alpha = 1.0f, beta = 0.0f;
    const int N_in = batch_size * seq_len;

    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V, embed_dim));
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    const bool is_causal = true;
    dim3 grid(batch_size * seq_len, num_heads);
    dim3 block(128);
    size_t shared_mem_size = 2 * 128 * head_dim * sizeof(float);
    fused_attention_forward_kernel<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_attn_heads_output, d_lse, batch_size, num_heads, seq_len, head_dim, scale_factor, is_causal);
    dim3 concat_blocks(N_in), concat_threads(32, 8);
    concat_heads_kernel<<<concat_blocks, concat_threads>>>(d_attn_heads_output, d_concat_heads, batch_size, num_heads, seq_len, head_dim);
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_concat_heads, embed_dim, &beta, d_output, embed_dim));
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, embed_dim, N_in, &alpha, d_grad_output, embed_dim, d_concat_heads, embed_dim, &beta, d_grad_Wo, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_grad_output, embed_dim, &beta, d_grad_concat_heads, embed_dim));
    split_heads_grad_kernel<<<concat_blocks, concat_threads>>>(d_grad_concat_heads, d_grad_attn_heads_output, batch_size, num_heads, seq_len, head_dim);
    
    CHECK_CUDA(cudaMemset(d_grad_Q, 0, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_K, 0, h_input.size()*sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_V, 0, h_input.size()*sizeof(float)));
    fused_attention_backward_kernel<<<grid, block, 4 * 128 * head_dim * sizeof(float)>>>(d_Q, d_K, d_V, d_output, d_lse, d_grad_attn_heads_output, d_grad_Q, d_grad_K, d_grad_V, batch_size, num_heads, seq_len, head_dim, scale_factor, is_causal);
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, N_in, &alpha, d_grad_Q, embed_dim, d_input, embed_dim, &beta, d_grad_Wq, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, N_in, &alpha, d_grad_K, embed_dim, d_input, embed_dim, &beta, d_grad_Wk, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, N_in, &alpha, d_grad_V, embed_dim, d_input, embed_dim, &beta, d_grad_Wv, embed_dim));
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_grad_Q, embed_dim, &beta, d_grad_input_from_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_grad_K, embed_dim, &beta, d_grad_input_from_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_grad_V, embed_dim, &beta, d_grad_input_from_V, embed_dim));
    add_matrices_kernel<<<(h_input.size() + 255) / 256, 256>>>(d_grad_input, d_grad_input_from_Q, d_grad_input_from_K, d_grad_input_from_V, h_input.size());

    print_matrix("Gradient w.r.t. Input", d_grad_input, N_in, embed_dim);


    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
