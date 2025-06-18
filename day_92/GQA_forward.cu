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
        fprintf(stderr, "cublasStatus: %d\n", status); \
        exit(1); \
    } \
}

// softmax & scaling kernels

__global__ void find_max_kernel(const float* input, float* max_vals, int N, int D) {
    int n = blockIdx.x;
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = -INFINITY;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], input[n * D + i]);
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_vals[n] = sdata[0];
}

__global__ void exp_sum_normalize_kernel(float* data, const float* max_vals, float* sum_vals, int N, int D) {
    int n = blockIdx.x;
    extern __shared__ float sdata[];
    float max_val = max_vals[n];
    sdata[threadIdx.x] = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = expf(data[n * D + i] - max_val);
        data[n * D + i] = val;
        sdata[threadIdx.x] += val;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sum_vals[n] = sdata[0];
    __syncthreads();
    float sum_val = sum_vals[n];
    if (sum_val > 0) {
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            data[n * D + i] /= sum_val;
        }
    }
}

__global__ void scale_kernel(float* matrix, float scale_factor, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        matrix[i] *= scale_factor;
    }
}

__global__ void concat_heads_kernel(const float* input, float* output, int N, int H, int L, int Dh) {
    int n = blockIdx.x / L;
    int l = blockIdx.x % L;
    
    for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
            int input_idx = n * H * L * Dh + h * L * Dh + l * Dh + d;
            int output_idx = n * L * (H * Dh) + l * (H * Dh) + h * Dh + d;
            output[output_idx] = input[input_idx];
        }
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
    const int batch_size = 32;       
    const int seq_len = 64;          
    const int embed_dim = 512;       
    const int num_q_heads = 8;         
    const int num_kv_heads = 2;        
    
    if (embed_dim % num_q_heads != 0 || num_q_heads % num_kv_heads != 0) {
        std::cerr << "Dimension or head configuration is invalid." << std::endl;
        return 1;
    }
    const int head_dim = embed_dim / num_q_heads; // Dimension of each head
    const int heads_per_group = num_q_heads / num_kv_heads;

    std::cout << "--- Grouped-Query Attention (GQA) Forward Pass ---" << std::endl;
    std::cout << "Query Heads: " << num_q_heads << ", KV Heads: " << num_kv_heads << ", Heads per Group: " << heads_per_group << std::endl << std::endl;

    std::vector<float> h_input(batch_size * seq_len * embed_dim);
    std::vector<float> h_Wq(embed_dim * embed_dim);
    std::vector<float> h_Wk(embed_dim * (num_kv_heads * head_dim));
    std::vector<float> h_Wv(embed_dim * (num_kv_heads * head_dim));
    std::vector<float> h_Wo(embed_dim * embed_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& v : h_input) v = dis(gen);
    for (float& v : h_Wq) v = dis(gen);
    for (float& v : h_Wk) v = dis(gen);
    for (float& v : h_Wv) v = dis(gen);
    for (float& v : h_Wo) v = dis(gen);

    float *d_input, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K_gqa, *d_V_gqa; 
    float *d_scores, *d_softmax_scores, *d_attn_heads_output, *d_concat_heads, *d_output;
    float *d_softmax_max, *d_softmax_sum;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq, h_Wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, h_Wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, h_Wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, h_Wo.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, batch_size * seq_len * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K_gqa, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V_gqa, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_heads_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_concat_heads, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_max, batch_size * num_q_heads * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_sum, batch_size * num_q_heads * seq_len * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo.data(), h_Wo.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f;
    const int N_in = batch_size * seq_len;
    const int kv_embed_dim = num_kv_heads * head_dim;

    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, kv_embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K_gqa, kv_embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, kv_embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V_gqa, kv_embed_dim));
    
    
    long long int k_head_stride = seq_len * head_dim;
    long long int q_head_stride = seq_len * head_dim;
    
    for (int i = 0; i < num_kv_heads; ++i) {
        const float* current_k_group = d_K_gqa + i * k_head_stride;
        const float* current_q_group = d_Q + i * heads_per_group * q_head_stride;
        float* current_scores_group = d_scores + i * heads_per_group * seq_len * seq_len;
        
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            seq_len, seq_len, head_dim,
            &alpha,
            current_k_group, head_dim, 0, 
            current_q_group, head_dim, q_head_stride, 
            &beta,
            current_scores_group, seq_len, seq_len * seq_len,
            batch_size * heads_per_group)); 
    }
    
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    scale_kernel<<<(batch_size * num_q_heads * seq_len * seq_len + 255) / 256, 256>>>(d_scores, scale_factor, batch_size * num_q_heads * seq_len * seq_len);
    
    CHECK_CUDA(cudaMemcpy(d_softmax_scores, d_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToDevice));
    int softmax_rows = batch_size * num_q_heads * seq_len;
    find_max_kernel<<<softmax_rows, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, softmax_rows, seq_len);
    exp_sum_normalize_kernel<<<softmax_rows, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, d_softmax_sum, softmax_rows, seq_len);
    
    for (int i = 0; i < num_kv_heads; ++i) {
        const float* current_v_group = d_V_gqa + i * k_head_stride; 
        const float* current_softmax_group = d_softmax_scores + i * heads_per_group * seq_len * seq_len;
        float* current_output_group = d_attn_heads_output + i * heads_per_group * q_head_stride;
        
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, seq_len, seq_len,
            &alpha,
            current_v_group, head_dim, 0, 
            current_softmax_group, seq_len, seq_len * seq_len, 
            &beta,
            current_output_group, head_dim, q_head_stride,
            batch_size * heads_per_group));
    }
    
    dim3 concat_blocks(N_in);
    dim3 concat_threads(32, 8);
    concat_heads_kernel<<<concat_blocks, concat_threads>>>(d_attn_heads_output, d_concat_heads, batch_size, num_q_heads, seq_len, head_dim);
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_concat_heads, embed_dim, &beta, d_output, embed_dim));

    std::cout << "GQA forward pass complete." << std::endl;
    print_matrix("Final Output", d_output, N_in, embed_dim);

    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
