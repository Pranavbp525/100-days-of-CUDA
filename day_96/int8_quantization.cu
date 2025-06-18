include <iostream>
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
        fprintf(stderr, "cuDNN status: %s\n", cudnnGetErrorString(status)); \
        exit(1); \
    } \
}




__global__ void find_abs_max_kernel(const float* input, float* output, int n_elements) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0.0f;
    while (i < n_elements) {
        sdata[tid] = fmaxf(sdata[tid], fabsf(input[i]));
        i += gridDim.x * blockDim.x;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(output, sdata[0]);
    }
}

__global__ void quantize_to_int8_kernel_gpu(const float* input, int8_t* output, const float* d_abs_max, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        // All threads read the same scale factor derived from the device pointer
        float scale = d_abs_max[0] / 127.0f;
        
        float clamped_val = fminf(fmaxf(input[i] / scale, -127.0f), 127.0f);
        output[i] = static_cast<int8_t>(roundf(clamped_val));
    }
}

__global__ void int8_gemm_kernel(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int32_t sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += static_cast<int32_t>(A[row * K + i]) * static_cast<int32_t>(B[i * N + col]);
        }
        C[row * N + col] = sum;
    }
}


__global__ void dequantize_to_fp32_kernel_gpu(const int32_t* input, float* output, const float* d_scale_A, const float* d_scale_B, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        float scale_A = d_scale_A[0] / 127.0f;
        float scale_B = d_scale_B[0] / 127.0f;
        output[i] = static_cast<float>(input[i]) * scale_A * scale_B;
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

float calculate_error(const float* ref, const float* quantized, int n_elements) {
    std::vector<float> h_ref(n_elements);
    std::vector<float> h_quantized(n_elements);
    CHECK_CUDA(cudaMemcpy(h_ref.data(), ref, n_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_quantized.data(), quantized, n_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    double total_error = 0.0;
    for(int i=0; i<n_elements; ++i) {
        total_error += std::fabs(h_ref[i] - h_quantized[i]);
    }
    return static_cast<float>(total_error / n_elements);
}


int main() {
    const int M = 512;
    const int N = 1024;
    const int K = 2048;

    std::vector<float> h_input(M * K);
    std::vector<float> h_weights(K * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for(float& v : h_input) v = dis(gen);
    for(float& v : h_weights) v = dis(gen);

    float *d_input, *d_weights, *d_ref_output, *d_quantized_output;
    int8_t *d_input_q, *d_weights_q;
    int32_t *d_gemm_output_q;
    float *d_abs_max_input, *d_abs_max_weights;
    
    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, h_weights.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ref_output, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_quantized_output, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_input_q, h_input.size() * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_weights_q, h_weights.size() * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_gemm_output_q, M * N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_abs_max_input, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_abs_max_weights, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_weights, N, d_input, K, &beta, d_ref_output, N));
    std::cout << "--- 1. Reference FP32 GEMM Complete ---" << std::endl;
    print_matrix("Reference FP32 Output", d_ref_output, M, N);


    CHECK_CUDA(cudaMemset(d_abs_max_input, 0, sizeof(float)));
    CHECK_CUDA(cudaMemset(d_abs_max_weights, 0, sizeof(float)));
    find_abs_max_kernel<<<256, 256, 256*sizeof(float)>>>(d_input, d_abs_max_input, M * K);
    find_abs_max_kernel<<<256, 256, 256*sizeof(float)>>>(d_weights, d_abs_max_weights, K * N);

    quantize_to_int8_kernel_gpu<<<(M * K + 255) / 256, 256>>>(d_input, d_input_q, d_abs_max_input, M * K);
    quantize_to_int8_kernel_gpu<<<(K * N + 255) / 256, 256>>>(d_weights, d_weights_q, d_abs_max_weights, K * N);
    std::cout << "--- 2. Quantization to INT8 Complete (on GPU) ---" << std::endl;

    dim3 grid( (N + 15) / 16, (M + 15) / 16 );
    dim3 block(16, 16);
    int8_gemm_kernel<<<grid, block>>>(d_input_q, d_weights_q, d_gemm_output_q, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "--- 3. Custom INT8 GEMM Complete ---" << std::endl;

    dequantize_to_fp32_kernel_gpu<<<(M * N + 255) / 256, 256>>>(d_gemm_output_q, d_quantized_output, d_abs_max_input, d_abs_max_weights, M * N);
    std::cout << "--- 4. Dequantization to FP32 Complete (on GPU) ---" << std::endl;
    print_matrix("Quantized FP32 Output", d_quantized_output, M, N);

    float avg_error = calculate_error(d_ref_output, d_quantized_output, M * N);
    std::cout << "--- 5. Comparison ---" << std::endl;
    std::cout << "Average Absolute Error: " << avg_error << std::endl;

    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_weights)); CHECK_CUDA(cudaFree(d_ref_output));
    CHECK_CUDA(cudaFree(d_quantized_output)); CHECK_CUDA(cudaFree(d_input_q)); CHECK_CUDA(cudaFree(d_weights_q));
    CHECK_CUDA(cudaFree(d_gemm_output_q)); CHECK_CUDA(cudaFree(d_abs_max_input)); CHECK_CUDA(cudaFree(d_abs_max_weights));

    return 0;
}
