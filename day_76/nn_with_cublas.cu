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




__global__ void layer_norm_forward_kernel(
    const float* input, float* output, const float* gamma, const float* beta,
    float* mean, float* rstd, int N, int D, float eps) {
    int n = blockIdx.x;
    extern __shared__ float shared_data[];

    float thread_sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        thread_sum += input[n * D + d];
    }
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) mean[n] = shared_data[0] / D;
    __syncthreads();

    float sample_mean = mean[n];
    float thread_sq_sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float diff = input[n * D + d] - sample_mean;
        thread_sq_sum += diff * diff;
    }
    shared_data[threadIdx.x] = thread_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        __syncthreads();
    }
    
    if (threadIdx.x == 0) rstd[n] = rsqrtf(shared_data[0] / D + eps);
    __syncthreads();

    float sample_rstd = rstd[n];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float normalized = (input[n * D + d] - sample_mean) * sample_rstd;
        output[n * D + d] = gamma[d] * normalized + beta[d];
    }
}

__global__ void layer_norm_backward_input_kernel(
    const float* d_out, const float* input, const float* gamma, const float* mean, const float* rstd,
    float* d_in, int N, int D) {
    // Each block processes one sample
    int n = blockIdx.x;
    extern __shared__ float shared_data[];
    
    float sample_rstd = rstd[n];
    float sample_mean = mean[n];

    float sum1 = 0;
    float sum2 = 0;
    // Each thread in the block contributes to the reduction for a single sample
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float normalized = (input[n * D + d] - sample_mean) * sample_rstd;
        sum1 += d_out[n * D + d] * gamma[d];
        sum2 += d_out[n * D + d] * gamma[d] * normalized;
    }
    shared_data[threadIdx.x] = sum1;
    (shared_data + blockDim.x)[threadIdx.x] = sum2;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
            (shared_data + blockDim.x)[threadIdx.x] += (shared_data + blockDim.x)[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum1 = shared_data[0];
    sum2 = (shared_data + blockDim.x)[0];
    
    // Each thread calculates the gradient for its assigned elements
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float normalized = (input[n * D + d] - sample_mean) * sample_rstd;
        float term1 = D * d_out[n * D + d] * gamma[d];
        float term2 = sum1;
        float term3 = normalized * sum2;
        d_in[n * D + d] = (1.0f / (D * sample_rstd)) * (term1 - term2 - term3);
    }
}

__global__ void layer_norm_param_gradients_kernel(
    const float* d_out, const float* input, const float* mean, const float* rstd,
    float* d_gamma, float* d_beta, int N, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < D) {
        float gamma_grad = 0.0f;
        float beta_grad = 0.0f;
        for (int n = 0; n < N; n++) {
            float normalized = (input[n * D + d] - mean[n]) * rstd[n];
            gamma_grad += d_out[n * D + d] * normalized;
            beta_grad += d_out[n * D + d];
        }
        d_gamma[d] = gamma_grad;
        d_beta[d] = beta_grad;
    }
}


// Softmax & Cross-Entropy Kernels
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
        data[n * D + i] = val; // Store intermediate exp value
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

__global__ void cross_entropy_loss_and_grad_kernel(const float* predictions, const int* labels, float* loss, float* d_in, int N, int C) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    
    sdata[tid] = 0.0f;
    if (i < N) {
        int target_idx = i * C + labels[i];
        sdata[tid] = -logf(fmaxf(predictions[target_idx], 1e-9f));
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(loss, sdata[0]);

    // Fused backward pass for Softmax + CrossEntropy
    if (i < N * C) {
        int n = i / C;
        int c = i % C;
        int target_class = labels[n];
        float grad = predictions[i] - (c == target_class ? 1.0f : 0.0f);
        d_in[i] = grad / N; // Average gradient over batch
    }
}

//Adam
__global__ void adam_update_kernel(
    float* params, const float* gradients, float* m, float* v, int size,
    float learning_rate, float beta1, float beta2, float epsilon, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gradients[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        params[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}


__global__ void add_bias_kernel(float* matrix, const float* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        matrix[row * cols + col] += bias[col];
    }
}

__global__ void relu_forward_kernel(float* matrix, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) matrix[i] = fmaxf(0.0f, matrix[i]);
}

__global__ void relu_backward_kernel(const float* pre_activation, float* incoming_grad, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements && pre_activation[i] <= 0) {
        incoming_grad[i] = 0;
    }
}

// Kernel to correctly compute bias gradients by summing rows of the incoming gradient matrix
__global__ void reduce_rows_kernel(const float* grad_in, float* grad_out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += grad_in[row * cols + col];
        }
        grad_out[col] = sum;
    }
}

int main() {
    const int epochs = 50;
    const int batch_size = 256;
    const int in_features = 784;
    const int hidden_features = 256;
    const int out_features = 10;

    const float learning_rate = 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;
    const float ln_eps = 1e-5f;

    std::vector<float> h_input(batch_size * in_features);
    std::vector<int> h_labels(batch_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < in_features; ++j) h_input[i * in_features + j] = dis(gen);
        h_labels[i] = gen() % out_features;
    }

    float std_dev1 = sqrtf(2.0f / (in_features + hidden_features));
    float std_dev2 = sqrtf(2.0f / (hidden_features + out_features));
    std::normal_distribution<float> d1(0.0f, std_dev1), d2(0.0f, std_dev2);
    std::vector<float> h_w1(in_features * hidden_features), h_w2(hidden_features * out_features);
    std::vector<float> h_b1(hidden_features, 0.0f), h_b2(out_features, 0.0f);
    std::vector<float> h_gamma(hidden_features, 1.0f), h_beta(hidden_features, 0.0f);
    for (float& v : h_w1) v = d1(gen);
    for (float& v : h_w2) v = d2(gen);
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    float *d_input, *d_w1, *d_b1, *d_w2, *d_b2, *d_gamma, *d_beta;
    int* d_labels;
    float *d_z1, *d_a1, *d_ln_out, *d_z2, *d_a2_softmax;
    float *d_gz2, *d_gw2, *d_gb2, *d_ga1, *d_gln_in, *d_ggamma, *d_gbeta, *d_gz1, *d_gw1, *d_gb1;
    float *d_ln_mean, *d_ln_rstd, *d_softmax_max, *d_softmax_sum, *d_loss;
    float *d_m_w1, *d_v_w1, *d_m_b1, *d_v_b1, *d_m_w2, *d_v_w2, *d_m_b2, *d_v_b2, *d_m_gamma, *d_v_gamma, *d_m_beta, *d_v_beta;

    CHECK_CUDA(cudaMalloc(&d_input, batch_size * in_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_labels, batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_w1, in_features * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b1, hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2, hidden_features * out_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b2, out_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gamma, hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z1, batch_size * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a1, batch_size * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_out, batch_size * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_z2, batch_size * out_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a2_softmax, batch_size * out_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gz2, batch_size * out_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gw2, hidden_features * out_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gb2, out_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ga1, batch_size * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gln_in, batch_size * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ggamma, hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gbeta, hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gz1, batch_size * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gw1, in_features * hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gb1, hidden_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_mean, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ln_rstd, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_max, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_softmax_sum, batch_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_w1, h_w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_w1, h_w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_b1, h_b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_b1, h_b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_w2, h_w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_w2, h_w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_b2, h_b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_b2, h_b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_gamma, h_gamma.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_gamma, h_gamma.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_beta, h_beta.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v_beta, h_beta.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma.data(), h_gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_m_w1, 0, h_w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_w1, 0, h_w1.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_m_b1, 0, h_b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_b1, 0, h_b1.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_m_w2, 0, h_w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_w2, 0, h_w2.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_m_b2, 0, h_b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_b2, 0, h_b2.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_m_gamma, 0, h_gamma.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_gamma, 0, h_gamma.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_m_beta, 0, h_beta.size() * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v_beta, 0, h_beta.size() * sizeof(float)));

    std::cout << "\n--- Starting Training Loop ---\n" << std::endl;
    const float alpha = 1.0f, beta = 0.0f;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // --- 1. Forward Pass ---
        // Z1(hidden, batch) = W1(hidden, in) @ Input(in, batch)
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_features, batch_size, in_features, &alpha, d_w1, hidden_features, d_input, in_features, &beta, d_z1, hidden_features));
        dim3 bias1_blocks((hidden_features + 15) / 16, (batch_size + 15) / 16);
        dim3 bias1_threads(16, 16);
        add_bias_kernel<<<bias1_blocks, bias1_threads>>>(d_z1, d_b1, batch_size, hidden_features);
        CHECK_CUDA(cudaMemcpy(d_a1, d_z1, batch_size * hidden_features * sizeof(float), cudaMemcpyDeviceToDevice));
        relu_forward_kernel<<<(batch_size * hidden_features + 255) / 256, 256>>>(d_a1, batch_size * hidden_features);
        layer_norm_forward_kernel<<<batch_size, 256, 256 * sizeof(float)>>>(d_a1, d_ln_out, d_gamma, d_beta, d_ln_mean, d_ln_rstd, batch_size, hidden_features, ln_eps);
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, batch_size, hidden_features, &alpha, d_w2, out_features, d_ln_out, hidden_features, &beta, d_z2, out_features));
        dim3 bias2_blocks((out_features + 15) / 16, (batch_size + 15) / 16);
        dim3 bias2_threads(16, 16);
        add_bias_kernel<<<bias2_blocks, bias2_threads>>>(d_z2, d_b2, batch_size, out_features);
        
        // Softmax
        CHECK_CUDA(cudaMemcpy(d_a2_softmax, d_z2, batch_size * out_features * sizeof(float), cudaMemcpyDeviceToDevice));
        find_max_kernel<<<batch_size, 256, 256 * sizeof(float)>>>(d_a2_softmax, d_softmax_max, batch_size, out_features);
        exp_sum_normalize_kernel<<<batch_size, 256, 256 * sizeof(float)>>>(d_a2_softmax, d_softmax_max, d_softmax_sum, batch_size, out_features);
        
        // --- 2. Loss & Initial Gradient ---
        CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
        cross_entropy_loss_and_grad_kernel<<<(batch_size * out_features + 255) / 256, 256, 256 * sizeof(float)>>>(d_a2_softmax, d_labels, d_loss, d_gz2, batch_size, out_features);
        
        // --- 3. Backward Pass ---
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, hidden_features, batch_size, &alpha, d_gz2, out_features, d_ln_out, hidden_features, &beta, d_gw2, out_features));
        reduce_rows_kernel<<<(out_features + 255) / 256, 256>>>(d_gz2, d_gb2, batch_size, out_features);
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, hidden_features, batch_size, out_features, &alpha, d_w2, out_features, d_gz2, out_features, &beta, d_ga1, hidden_features));
        CHECK_CUDA(cudaMemset(d_ggamma, 0, hidden_features * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_gbeta, 0, hidden_features * sizeof(float)));
        
        // Corrected LayerNorm Backward
        layer_norm_backward_input_kernel<<<batch_size, 256, 2 * 256 * sizeof(float)>>>(d_ga1, d_a1, d_gamma, d_ln_mean, d_ln_rstd, d_gln_in, batch_size, hidden_features);
        layer_norm_param_gradients_kernel<<<(hidden_features + 255) / 256, 256>>>(d_ga1, d_a1, d_ln_mean, d_ln_rstd, d_ggamma, d_gbeta, batch_size, hidden_features);

        relu_backward_kernel<<<(batch_size * hidden_features + 255) / 256, 256>>>(d_z1, d_gln_in, batch_size * hidden_features);
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, hidden_features, batch_size, &alpha, d_input, in_features, d_gln_in, hidden_features, &beta, d_gw1, in_features));
        reduce_rows_kernel<<<(hidden_features + 255) / 256, 256>>>(d_gln_in, d_gb1, batch_size, hidden_features);

        // --- 4. Optimizer Step ---
        adam_update_kernel<<<(h_w1.size() + 255) / 256, 256>>>(d_w1, d_gw1, d_m_w1, d_v_w1, h_w1.size(), learning_rate, beta1, beta2, epsilon, epoch);
        adam_update_kernel<<<(h_b1.size() + 255) / 256, 256>>>(d_b1, d_gb1, d_m_b1, d_v_b1, h_b1.size(), learning_rate, beta1, beta2, epsilon, epoch);
        adam_update_kernel<<<(h_w2.size() + 255) / 256, 256>>>(d_w2, d_gw2, d_m_w2, d_v_w2, h_w2.size(), learning_rate, beta1, beta2, epsilon, epoch);
        adam_update_kernel<<<(h_b2.size() + 255) / 256, 256>>>(d_b2, d_gb2, d_m_b2, d_v_b2, h_b2.size(), learning_rate, beta1, beta2, epsilon, epoch);
        adam_update_kernel<<<(h_gamma.size() + 255) / 256, 256>>>(d_gamma, d_ggamma, d_m_gamma, d_v_gamma, h_gamma.size(), learning_rate, beta1, beta2, epsilon, epoch);
        adam_update_kernel<<<(h_beta.size() + 255) / 256, 256>>>(d_beta, d_gbeta, d_m_beta, d_v_beta, h_beta.size(), learning_rate, beta1, beta2, epsilon, epoch);

        if (epoch % 5 == 0 || epoch == 1) {
            float h_loss;
            CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "Epoch " << epoch << ", Loss: " << h_loss / batch_size << std::endl;
        }
    }

    std::cout << "\n--- Training Complete & Cleaning up resources ---" << std::endl;
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_labels)); CHECK_CUDA(cudaFree(d_w1)); CHECK_CUDA(cudaFree(d_b1)); CHECK_CUDA(cudaFree(d_w2)); CHECK_CUDA(cudaFree(d_b2)); CHECK_CUDA(cudaFree(d_gamma)); CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(d_z1)); CHECK_CUDA(cudaFree(d_a1)); CHECK_CUDA(cudaFree(d_ln_out)); CHECK_CUDA(cudaFree(d_z2)); CHECK_CUDA(cudaFree(d_a2_softmax));
    CHECK_CUDA(cudaFree(d_gz2)); CHECK_CUDA(cudaFree(d_gw2)); CHECK_CUDA(cudaFree(d_gb2)); CHECK_CUDA(cudaFree(d_ga1)); CHECK_CUDA(cudaFree(d_gln_in)); CHECK_CUDA(cudaFree(d_ggamma)); CHECK_CUDA(cudaFree(d_gbeta)); CHECK_CUDA(cudaFree(d_gz1)); CHECK_CUDA(cudaFree(d_gw1)); CHECK_CUDA(cudaFree(d_gb1));
    CHECK_CUDA(cudaFree(d_ln_mean)); CHECK_CUDA(cudaFree(d_ln_rstd)); CHECK_CUDA(cudaFree(d_softmax_max)); CHECK_CUDA(cudaFree(d_softmax_sum)); CHECK_CUDA(cudaFree(d_loss));
    CHECK_CUDA(cudaFree(d_m_w1)); CHECK_CUDA(cudaFree(d_v_w1)); CHECK_CUDA(cudaFree(d_m_b1)); CHECK_CUDA(cudaFree(d_v_b1)); CHECK_CUDA(cudaFree(d_m_w2)); CHECK_CUDA(cudaFree(d_v_w2)); CHECK_CUDA(cudaFree(d_m_b2)); CHECK_CUDA(cudaFree(d_v_b2));
    CHECK_CUDA(cudaFree(d_m_gamma)); CHECK_CUDA(cudaFree(d_v_gamma)); CHECK_CUDA(cudaFree(d_m_beta)); CHECK_CUDA(cudaFree(d_v_beta));
    
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    return 0;
}
