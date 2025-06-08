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

// softmax kernels

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
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        data[n * D + i] /= sum_val;
    }
}

__global__ void scale_kernel(float* matrix, float scale_factor, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        matrix[i] *= scale_factor;
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
    // Attention Layer Dimensions
    const int batch_size = 32;       // N
    const int seq_len = 64;          // L (Sequence Length)
    const int embed_dim = 512;       // D (Embedding Dimension)
    const int d_k = embed_dim;       // Dimension of Key/Query, must match embed_dim for this simple case

    std::cout << "--- Self-Attention Forward Pass ---" << std::endl;
    std::cout << "Batch: " << batch_size << ", Seq Len: " << seq_len << ", Embed Dim: " << embed_dim << std::endl << std::endl;

    std::vector<float> h_input(batch_size * seq_len * embed_dim);
    std::vector<float> h_Wq(embed_dim * d_k);
    std::vector<float> h_Wk(embed_dim * d_k);
    std::vector<float> h_Wv(embed_dim * embed_dim); // Wv maps to the same dimension

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& v : h_input) v = dis(gen);
    for (float& v : h_Wq) v = dis(gen);
    for (float& v : h_Wk) v = dis(gen);
    for (float& v : h_Wv) v = dis(gen);

    float *d_input, *d_Wq, *d_Wk, *d_Wv;
    float *d_Q, *d_K, *d_V;
    float *d_scores, *d_softmax_scores, *d_output;
    float *d_softmax_max, *d_softmax_sum;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq, h_Wq.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, h_Wk.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, h_Wv.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, batch_size * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_scores, batch_size * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_max, batch_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_sum, batch_size * seq_len * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq.data(), h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk.data(), h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv.data(), h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f;
    const int N_in = batch_size * seq_len; // Treat the batched sequence as one large matrix for GEMM

    

    // Q = Input @ Wq
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, d_k, N_in, embed_dim, &alpha, d_Wq, embed_dim, d_input, embed_dim, &beta, d_Q, d_k));
    // K = Input @ Wk
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, d_k, N_in, embed_dim, &alpha, d_Wk, embed_dim, d_input, embed_dim, &beta, d_K, d_k));
    // V = Input @ Wv
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, embed_dim, N_in, embed_dim, &alpha, d_Wv, embed_dim, d_input, embed_dim, &beta, d_V, embed_dim));
    
    // Calculate Attention Scores: scores = Q @ K^T using batched GEMM

    // 1 sample in batch: (seq_len x d_k) @ (d_k x seq_len) -> (seq_len x seq_len) 
    CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,       // K^T, Q
        seq_len, seq_len, d_k,         // m, n, k
        &alpha,
        d_K, d_k, seq_len * d_k,       // A (K)
        d_Q, d_k, seq_len * d_k,       // B (Q)
        &beta,
        d_scores, seq_len, seq_len * seq_len, // C (Scores)
        batch_size));                  // batchCount

    // Scale Scores
    const float scale_factor = 1.0f / sqrtf((float)d_k);
    scale_kernel<<<(batch_size * seq_len * seq_len + 255) / 256, 256>>>(d_scores, scale_factor, batch_size * seq_len * seq_len);

    // Softmax 
    CHECK_CUDA(cudaMemcpy(d_softmax_scores, d_scores, batch_size * seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToDevice));
    // Each "row" for softmax is a row of the attention score matrix. There are (batch_size * seq_len) such rows.
    find_max_kernel<<<batch_size * seq_len, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, batch_size * seq_len, seq_len);
    exp_sum_normalize_kernel<<<batch_size * seq_len, 256, 256 * sizeof(float)>>>(d_softmax_scores, d_softmax_max, d_softmax_sum, batch_size * seq_len, seq_len);
    
    // output = Softmax_Scores @ V
    // Batched GEMM: (seq_len x seq_len) @ (seq_len x embed_dim) -> (seq_len x embed_dim)
    CHECK_CUBLAS(cublasSgemmStridedBatched(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        embed_dim, seq_len, seq_len,    // m, n, k
        &alpha,
        d_V, embed_dim, seq_len * embed_dim, // A (V)
        d_softmax_scores, seq_len, seq_len * seq_len, // B (Scores)
        &beta,
        d_output, embed_dim, seq_len * embed_dim, // C (Output)
        batch_size));

    std::cout << "Attention computation complete." << std::endl << std::endl;
    print_matrix("Final Output", d_output, batch_size * seq_len, embed_dim);

    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_Wq)); CHECK_CUDA(cudaFree(d_Wk)); CHECK_CUDA(cudaFree(d_Wv));
    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_K)); CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_scores)); CHECK_CUDA(cudaFree(d_softmax_scores)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_softmax_max)); CHECK_CUDA(cudaFree(d_softmax_sum));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
