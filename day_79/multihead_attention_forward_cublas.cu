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

// Reshaping Head Outputs
// This kernel takes the interleaved output of the attention heads (N, H, L, Dh)
// and reshapes it to a contiguous format (N, L, D) where D = H * Dh.
__global__ void concat_heads_kernel(const float* input, float* output, int N, int H, int L, int Dh) {
    int n = blockIdx.x / L; // Batch index
    int l = blockIdx.x % L; // Sequence index
    
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
    // Multi-Head Attention Dimensions
    const int batch_size = 32;       // N
    const int seq_len = 64;          // L
    const int embed_dim = 512;       // D
    const int num_heads = 8;         // H
    
    if (embed_dim % num_heads != 0) {
        std::cerr << "Embedding dimension must be divisible by the number of heads." << std::endl;
        return 1;
    }
    const int head_dim = embed_dim / num_heads; // Dh

    std::cout << "--- Multi-Head Attention Forward Pass ---" << std::endl;
    std::cout << "Batch: " << batch_size << ", Seq Len: " << seq_len << ", Embed Dim: " << embed_dim 
              << ", Heads: " << num_heads << ", Head Dim: " << head_dim << std::endl << std::endl;

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
    float *d_scores, *d_softmax_scores, *d_attn_heads_output, *d_concat_heads, *d_output;
    float *d_softmax_max, *d_softmax_sum;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq, h_Wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, h_Wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, h_Wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, h_Wo.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_heads_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_concat_heads, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_max, batch_size * num_heads * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_sum, batch_size * num_heads * seq_len * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo.data(), h_Wo.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f;
    const int N_in = batch_size * seq_len;

    // Project Inputs to Q, K, V 
    // Q(N_in, D) = Input(N_in, D) @ Wq(D, D)
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K, embed_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V, embed_dim));
    
    // Calculate Attention Scores: scores = Q @ K^T 
    // (N, H, L, Dh) tensors as batched matrices.
    // Batch count = N * H. Matrix A = K (L, Dh), Matrix B = Q (L, Dh).
    // Result C = Scores (L, L).
    CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,       // K^T, Q
        seq_len, seq_len, head_dim,    // m, n, k
        &alpha,
        d_K, head_dim, seq_len * head_dim, // A (K) with its stride
        d_Q, head_dim, seq_len * head_dim, // B (Q) with its stride
        &beta,
        d_scores, seq_len, seq_len * seq_len, // C (Scores) with its stride
        batch_size * num_heads));            // batchCount

    // Scale Scores 
    const float scale_factor = 1.0f / sqrtf((float)head_dim);
    scale_kernel<<<(batch_size * num_heads * seq_len * seq_len + 255) / 256, 256>>>(d_scores, scale_factor, batch_size * num_heads * seq_len * seq_len);

    // Softmax 
    CHECK_CUDA(cudaMemcpy(d_softmax_scores, d_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToDevice));
    // Each row for softmax is a row of an attention score matrix. There are (batch_size * num_heads * seq_len) such rows.
    int softmax_rows = batch_size * num_heads * seq_len;
    int softmax_cols = seq_len;
    find_max_kernel<<<softmax_rows, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, softmax_rows, softmax_cols);
    exp_sum_normalize_kernel<<<softmax_rows, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, d_softmax_sum, softmax_rows, softmax_cols);
    
    // Apply Attention to V: attn_heads_output = Softmax_Scores @ V 
    // Batch count = N * H. Matrix A = V (L, Dh), Matrix B = Scores (L, L).
    // Result C = attn_heads_output (L, Dh).
    CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, seq_len, seq_len,    // m, n, k
        &alpha,
        d_V, head_dim, seq_len * head_dim, // A (V) with its stride
        d_softmax_scores, seq_len, seq_len * seq_len, // B (Scores) with its stride
        &beta,
        d_attn_heads_output, head_dim, seq_len * head_dim, // C (Output) with its stride
        batch_size * num_heads));

    // Concatenate Heads
    // Reshape from (N, H, L, Dh) to (N, L, D) where D = H * Dh
    dim3 concat_blocks(batch_size * seq_len);
    dim3 concat_threads(32, 8); // 256 threads total
    concat_heads_kernel<<<concat_blocks, concat_threads>>>(d_attn_heads_output, d_concat_heads, batch_size, num_heads, seq_len, head_dim);

    // Final Linear Projection 
    // Output(N*L, D) = Concat_Heads(N*L, D) @ Wo(D, D)
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wo, embed_dim, d_concat_heads, embed_dim, &beta, d_output, embed_dim));

    std::cout << "Multi-Head Attention forward pass complete." << std::endl << std::endl;
    print_matrix("Final Output", d_output, N_in, embed_dim);

    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_Wq)); CHECK_CUDA(cudaFree(d_Wk)); CHECK_CUDA(cudaFree(d_Wv)); CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_K)); CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_scores)); CHECK_CUDA(cudaFree(d_softmax_scores)); CHECK_CUDA(cudaFree(d_attn_heads_output)); CHECK_CUDA(cudaFree(d_concat_heads)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_softmax_max)); CHECK_CUDA(cudaFree(d_softmax_sum));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
