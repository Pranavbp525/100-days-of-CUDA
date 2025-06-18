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

__global__ void split_heads_grad_kernel(const float* grad_in, float* grad_out, int N, int H, int L, int Dh) {
    int n = blockIdx.x / L;
    int l = blockIdx.x % L;
    
    for (int h = threadIdx.y; h < H; h += blockDim.y) {
        for (int d = threadIdx.x; d < Dh; d += blockDim.x) {
            int input_idx = n * L * (H * Dh) + l * (H * Dh) + h * Dh + d;
            int output_idx = n * H * L * Dh + h * L * Dh + l * Dh + d;
            grad_out[output_idx] = grad_in[input_idx];
        }
    }
}

__global__ void softmax_backward_kernel(float* d_scores, const float* softmax_output, const float* d_softmax_scores, int N, int D) {
    int n = blockIdx.x;
    extern __shared__ float sdata[];

    float dot_product = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dot_product += d_softmax_scores[n * D + i] * softmax_output[n * D + i];
    }
    sdata[threadIdx.x] = dot_product;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    dot_product = sdata[0];
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        int idx = n * D + i;
        d_scores[idx] = softmax_output[idx] * (d_softmax_scores[idx] - dot_product);
    }
}

__global__ void add_matrices_kernel(float* out, const float* in1, const float* in2, const float* in3, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        out[i] = in1[i] + in2[i] + in3[i];
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
    const int head_dim = embed_dim / num_q_heads;
    const int heads_per_group = num_q_heads / num_kv_heads;

    
    std::vector<float> h_input(batch_size * seq_len * embed_dim);
    std::vector<float> h_Wq(embed_dim * embed_dim);
    std::vector<float> h_Wk(embed_dim * (num_kv_heads * head_dim));
    std::vector<float> h_Wv(embed_dim * (num_kv_heads * head_dim));
    std::vector<float> h_Wo(embed_dim * embed_dim);
    std::vector<float> h_grad_output(h_input.size());

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& v : h_input) v = dis(gen);
    for (float& v : h_Wq) v = dis(gen); for (float& v : h_Wk) v = dis(gen);
    for (float& v : h_Wv) v = dis(gen); for (float& v : h_Wo) v = dis(gen);
    for (float& v : h_grad_output) v = dis(gen);

    float *d_input, *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_Q, *d_K_gqa, *d_V_gqa;
    float *d_scores, *d_softmax_scores, *d_attn_heads_output, *d_concat_heads, *d_output;
    float *d_softmax_max, *d_softmax_sum;
    float *d_grad_output, *d_grad_concat_heads, *d_grad_Wo, *d_grad_attn_heads_output;
    float *d_grad_V_gqa, *d_grad_softmax_scores, *d_grad_scores, *d_grad_Q, *d_grad_K_gqa;
    float *d_grad_Wq, *d_grad_Wk, *d_grad_Wv, *d_grad_input;
    float *d_grad_input_from_Q, *d_grad_input_from_K, *d_grad_input_from_V;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq, h_Wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, h_Wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, h_Wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, h_Wo.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K_gqa, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V_gqa, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_heads_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_concat_heads, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_max, batch_size * num_q_heads * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_sum, batch_size * num_q_heads * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_concat_heads, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_Wo, h_Wo.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_attn_heads_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_V_gqa, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_softmax_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_Q, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_K_gqa, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_Wq, h_Wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_Wk, h_Wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_Wv, h_Wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input_from_Q, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input_from_K, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input_from_V, h_input.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo.data(), h_Wo.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f, beta_one = 1.0f;
    const int N_in = batch_size * seq_len;
    const int kv_embed_dim = num_kv_heads * head_dim;

    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, kv_embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K_gqa, kv_embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, kv_embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V_gqa, kv_embed_dim));
    
    long long int kv_head_stride = seq_len * head_dim;
    long long int q_head_stride = seq_len * head_dim;
    for (int i = 0; i < num_kv_heads; ++i) {
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim, &alpha, d_K_gqa + i * kv_head_stride, head_dim, 0, d_Q + i * heads_per_group * q_head_stride, head_dim, q_head_stride, &beta, d_scores + i * heads_per_group * seq_len * seq_len, seq_len, seq_len * seq_len, batch_size * heads_per_group));
    }
    
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    scale_kernel<<<(batch_size * num_q_heads * seq_len * seq_len + 255) / 256, 256>>>(d_scores, scale_factor, batch_size * num_q_heads * seq_len * seq_len);
    
    CHECK_CUDA(cudaMemcpy(d_softmax_scores, d_scores, batch_size * num_q_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToDevice));
    int softmax_rows = batch_size * num_q_heads * seq_len;
    find_max_kernel<<<softmax_rows, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, softmax_rows, seq_len);
    exp_sum_normalize_kernel<<<softmax_rows, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, d_softmax_sum, softmax_rows, seq_len);
    
    for (int i = 0; i < num_kv_heads; ++i) {
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len, &alpha, d_V_gqa + i * kv_head_stride, head_dim, 0, d_softmax_scores + i * heads_per_group * seq_len * seq_len, seq_len, seq_len * seq_len, &beta, d_attn_heads_output + i * heads_per_group * q_head_stride, head_dim, q_head_stride, batch_size * heads_per_group));
    }
    
    dim3 concat_blocks(N_in), concat_threads(32, 8);
    concat_heads_kernel<<<concat_blocks, concat_threads>>>(d_attn_heads_output, d_concat_heads, batch_size, num_q_heads, seq_len, head_dim);
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_concat_heads, embed_dim, &beta, d_output, embed_dim));

    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, embed_dim, N_in, &alpha, d_grad_output, embed_dim, d_concat_heads, embed_dim, &beta, d_grad_Wo, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_grad_output, embed_dim, &beta, d_grad_concat_heads, embed_dim));
    split_heads_grad_kernel<<<concat_blocks, concat_threads>>>(d_grad_concat_heads, d_grad_attn_heads_output, batch_size, num_q_heads, seq_len, head_dim);

    CHECK_CUDA(cudaMemset(d_grad_V_gqa, 0, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_grad_K_gqa, 0, batch_size * seq_len * num_kv_heads * head_dim * sizeof(float)));

    for (int i = 0; i < num_kv_heads; ++i) {
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, seq_len, seq_len, head_dim, &alpha, d_grad_attn_heads_output + i * heads_per_group * q_head_stride, head_dim, q_head_stride, d_V_gqa + i * kv_head_stride, head_dim, 0, &beta, d_grad_softmax_scores + i * heads_per_group * seq_len * seq_len, seq_len, seq_len*seq_len, batch_size * heads_per_group));
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, head_dim, seq_len, seq_len, &alpha, d_grad_attn_heads_output + i * heads_per_group * q_head_stride, head_dim, q_head_stride, d_softmax_scores + i * heads_per_group * seq_len * seq_len, seq_len, seq_len*seq_len, &beta_one, d_grad_V_gqa + i * kv_head_stride, head_dim, kv_head_stride, batch_size * heads_per_group));
    }
    
    softmax_backward_kernel<<<softmax_rows, 256, 256*sizeof(float)>>>(d_grad_scores, d_softmax_scores, d_grad_softmax_scores, softmax_rows, seq_len);
    scale_kernel<<<(batch_size * num_q_heads * seq_len * seq_len + 255) / 256, 256>>>(d_grad_scores, scale_factor, batch_size * num_q_heads * seq_len * seq_len);
    
    for (int i = 0; i < num_kv_heads; ++i) {
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, seq_len, seq_len, &alpha, d_K_gqa + i * kv_head_stride, head_dim, 0, d_grad_scores + i * heads_per_group * seq_len * seq_len, seq_len, seq_len*seq_len, &beta, d_grad_Q + i * heads_per_group * q_head_stride, head_dim, q_head_stride, batch_size * heads_per_group));
        CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, head_dim, seq_len, seq_len, &alpha, d_Q + i * heads_per_group * q_head_stride, head_dim, q_head_stride, d_grad_scores + i * heads_per_group * seq_len * seq_len, seq_len, seq_len*seq_len, &beta_one, d_grad_K_gqa + i * kv_head_stride, head_dim, kv_head_stride, batch_size * heads_per_group));
    }
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, embed_dim, N_in, &alpha, d_grad_Q, embed_dim, d_input, embed_dim, &beta, d_grad_Wq, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kv_embed_dim, embed_dim, N_in, &alpha, d_grad_K_gqa, kv_embed_dim, d_input, embed_dim, &beta, d_grad_Wk, kv_embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, kv_embed_dim, embed_dim, N_in, &alpha, d_grad_V_gqa, kv_embed_dim, d_input, embed_dim, &beta, d_grad_Wv, kv_embed_dim));
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_grad_Q, embed_dim, &beta, d_grad_input_from_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, kv_embed_dim, &alpha, d_Wk, embed_dim, d_grad_K_gqa, kv_embed_dim, &beta, d_grad_input_from_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, N_in, kv_embed_dim, &alpha, d_Wv, embed_dim, d_grad_V_gqa, kv_embed_dim, &beta, d_grad_input_from_V, embed_dim));
    add_matrices_kernel<<<(h_input.size() + 255) / 256, 256>>>(d_grad_input, d_grad_input_from_Q, d_grad_input_from_K, d_grad_input_from_V, h_input.size());

    std::cout << "Backward pass complete." << std::endl;
    print_matrix("Gradient w.r.t. Wq", d_grad_Wq, embed_dim, embed_dim);
    print_matrix("Gradient w.r.t. Wk", d_grad_Wk, embed_dim, kv_embed_dim);
    print_matrix("Gradient w.r.t. Input", d_grad_input, N_in, embed_dim);


    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
