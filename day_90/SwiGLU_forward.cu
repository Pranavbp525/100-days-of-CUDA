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


__global__ void swiglu_kernel(float* gated_projection, const float* gate_values, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_elements) {
        float x = gated_projection[i];
        
        float swish_x = x / (1.0f + expf(-x));
        
        gated_projection[i] = swish_x * gate_values[i];
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
    
    const int batch_size = 4;
    const int seq_len = 64;
    const int input_dim = 512;    
    const int hidden_dim = 1024;  

    const int total_tokens = batch_size * seq_len;

    std::cout << "--- SwiGLU Activation Forward Pass ---" << std::endl;
    std::cout << "Input Dim: " << input_dim << ", Hidden Dim: " << hidden_dim << std::endl << std::endl;

    std::vector<float> h_input(total_tokens * input_dim);
    std::vector<float> h_W(input_dim * hidden_dim);
    std::vector<float> h_V(input_dim * hidden_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& val : h_input) val = dis(gen);
    for (float& val : h_W) val = dis(gen);
    for (float& val : h_V) val = dis(gen);

    float *d_input, *d_W, *d_V;
    float *d_gated_projection, *d_gate_values;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W, h_W.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_V.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gated_projection, total_tokens * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate_values, total_tokens * hidden_dim * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f;


    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, hidden_dim, total_tokens, input_dim, &alpha, d_W, input_dim, d_input, input_dim, &beta, d_gated_projection, hidden_dim));
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, hidden_dim, total_tokens, input_dim, &alpha, d_V, input_dim, d_input, input_dim, &beta, d_gate_values, hidden_dim));

    std::cout << "Computed linear projections." << std::endl;
    print_matrix("Gated Projection (before SwiGLU)", d_gated_projection, total_tokens, hidden_dim);
    print_matrix("Gate Values", d_gate_values, total_tokens, hidden_dim);

    int n_elements = total_tokens * hidden_dim;
    dim3 swiglu_blocks((n_elements + 255) / 256);
    dim3 swiglu_threads(256);
    swiglu_kernel<<<swiglu_blocks, swiglu_threads>>>(d_gated_projection, d_gate_values, n_elements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << "Applied SwiGLU activation." << std::endl;
    print_matrix("Final SwiGLU Output", d_gated_projection, total_tokens, hidden_dim);
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_gated_projection));
    CHECK_CUDA(cudaFree(d_gate_values));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
