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

// swiglu activation kernels
__global__ void swiglu_forward_kernel(float* output, const float* gated_projection, const float* gate_values, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_elements) {
        float x = gated_projection[i];
        float swish_x = x / (1.0f + expf(-x));
        output[i] = swish_x * gate_values[i];
    }
}


__global__ void swiglu_backward_kernel(
    float* d_gated_projection, float* d_gate_values, 
    const float* d_output, const float* gated_projection, const float* gate_values, 
    int n_elements) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_elements) {
        float x = gated_projection[i];
        float g = gate_values[i];
        float d_out = d_output[i];

        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float d_swish = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));

        d_gated_projection[i] = d_out * g * d_swish;
        d_gate_values[i] = d_out * (x * sigmoid_x);
    }
}

__global__ void add_gradients_kernel(float* out, const float* in1, const float* in2, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        out[i] = in1[i] + in2[i];
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

    std::cout << "--- SwiGLU Activation Forward & Backward Pass ---" << std::endl;

    std::vector<float> h_input(total_tokens * input_dim);
    std::vector<float> h_W(input_dim * hidden_dim);
    std::vector<float> h_V(input_dim * hidden_dim);
    std::vector<float> h_grad_output(total_tokens * hidden_dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.02f);
    for (float& val : h_input) val = dis(gen);
    for (float& val : h_W) val = dis(gen);
    for (float& val : h_V) val = dis(gen);
    for (float& val : h_grad_output) val = dis(gen);

    float *d_input, *d_W, *d_V;
    float *d_gated_projection, *d_gate_values, *d_output;
    float *d_grad_output, *d_grad_gated_projection, *d_grad_gate_values;
    float *d_grad_W, *d_grad_V, *d_grad_input;
    float *d_grad_input_from_W, *d_grad_input_from_V;

    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W, h_W.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, h_V.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gated_projection, total_tokens * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate_values, total_tokens * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, total_tokens * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_output, h_grad_output.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_gated_projection, total_tokens * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_gate_values, total_tokens * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_W, h_W.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_V, h_V.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input_from_W, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_input_from_V, h_input.size() * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), h_W.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_grad_output, h_grad_output.data(), h_grad_output.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const float alpha = 1.0f, beta = 0.0f;

    std::cout << "\n--- 1. Forward Pass ---\n" << std::endl;
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, hidden_dim, total_tokens, input_dim, &alpha, d_W, input_dim, d_input, input_dim, &beta, d_gated_projection, hidden_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, hidden_dim, total_tokens, input_dim, &alpha, d_V, input_dim, d_input, input_dim, &beta, d_gate_values, hidden_dim));
    int n_elements = total_tokens * hidden_dim;
    swiglu_forward_kernel<<<(n_elements + 255) / 256, 256>>>(d_output, d_gated_projection, d_gate_values, n_elements);
    CHECK_CUDA(cudaDeviceSynchronize());
    print_matrix("Final SwiGLU Output", d_output, total_tokens, hidden_dim);
    
    std::cout << "\n--- 2. Backward Pass ---\n" << std::endl;
    swiglu_backward_kernel<<<(n_elements + 255) / 256, 256>>>(d_grad_gated_projection, d_grad_gate_values, d_grad_output, d_gated_projection, d_gate_values, n_elements);
    CHECK_CUDA(cudaDeviceSynchronize());
    print_matrix("Gradient w.r.t Gated Projection", d_grad_gated_projection, total_tokens, hidden_dim);

    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_dim, input_dim, total_tokens, &alpha, d_grad_gated_projection, hidden_dim, d_input, input_dim, &beta, d_grad_W, hidden_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, hidden_dim, input_dim, total_tokens, &alpha, d_grad_gate_values, hidden_dim, d_input, input_dim, &beta, d_grad_V, hidden_dim));
    
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, input_dim, total_tokens, hidden_dim, &alpha, d_W, input_dim, d_grad_gated_projection, hidden_dim, &beta, d_grad_input_from_W, input_dim));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, input_dim, total_tokens, hidden_dim, &alpha, d_V, input_dim, d_grad_gate_values, hidden_dim, &beta, d_grad_input_from_V, input_dim));

    add_gradients_kernel<<<(total_tokens * input_dim + 255) / 256, 256>>>(d_grad_input, d_grad_input_from_W, d_grad_input_from_V, total_tokens * input_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    print_matrix("Gradient w.r.t. W", d_grad_W, input_dim, hidden_dim);
    print_matrix("Gradient w.r.t. Input", d_grad_input, total_tokens, input_dim);
    
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_W)); CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_gated_projection)); CHECK_CUDA(cudaFree(d_gate_values)); CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_grad_output)); CHECK_CUDA(cudaFree(d_grad_gated_projection)); CHECK_CUDA(cudaFree(d_grad_gate_values));
    CHECK_CUDA(cudaFree(d_grad_W)); CHECK_CUDA(cudaFree(d_grad_V)); CHECK_CUDA(cudaFree(d_grad_input));
    CHECK_CUDA(cudaFree(d_grad_input_from_W)); CHECK_CUDA(cudaFree(d_grad_input_from_V));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
