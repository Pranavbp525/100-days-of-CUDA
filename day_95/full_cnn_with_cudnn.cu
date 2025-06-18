#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUDNN(call) { \
    const cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "cuDNN status: %s\n", cudnnGetErrorString(status)); \
        exit(1); \
    } \
}


void print_tensor(const std::string& name, const float* m, int N, int C, int H, int W) {
    std::cout << "--- " << name << " --- (N" << N << ", C" << C << ", H" << H << ", W" << W << ")\n";
    std::vector<float> h_m(N * C * H * W);
    CHECK_CUDA(cudaMemcpy(h_m.data(), m, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < std::min(8, H); ++i) {
        for (int j = 0; j < std::min(8, W); ++j) {
            std::cout << h_m[i * W + j] << "\t";
        }
        std::cout << (W > 8 ? "...\n" : "\n");
    }
    std::cout << (H > 8 ? "...\n" : "");
    std::cout << "-----------------------\n" << std::endl;
}


int main() {
    const int N = 4; const int C = 3; const int H = 256; const int W = 256;
    const int K = 16; const int R = 3; const int S = 3;
    const float alpha = 1.0f, beta = 0.0f;

    std::vector<float> h_input(N * C * H * W);
    std::vector<float> h_filter(K * C * R * S);
    for(size_t i = 0; i < h_input.size(); ++i) h_input[i] = (i % 256) / 255.0f;
    for(size_t i = 0; i < h_filter.size(); ++i) h_filter[i] = 0.5f;

    float *d_input, *d_filter;
    CHECK_CUDA(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_filter, h_filter.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter.data(), h_filter.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));

    std::cout << "--- 1. Convolution Step ---" << std::endl;
    cudnnTensorDescriptor_t input_desc, conv_output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv_output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    
    int conv_out_N, conv_out_C, conv_out_H, conv_out_W;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &conv_out_N, &conv_out_C, &conv_out_H, &conv_out_W));
    
    float* d_conv_output;
    CHECK_CUDA(cudaMalloc(&d_conv_output, conv_out_N * conv_out_C * conv_out_H * conv_out_W * sizeof(float)));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, conv_out_N, conv_out_C, conv_out_H, conv_out_W));
    
    cudnnConvolutionFwdAlgoPerf_t perf_results[1];
    int returned_algo_count;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle, input_desc, filter_desc, conv_desc, conv_output_desc, 1, &returned_algo_count, perf_results));
    cudnnConvolutionFwdAlgo_t conv_algo = perf_results[0].algo;
    
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, filter_desc, conv_desc, conv_output_desc, conv_algo, &workspace_bytes));
    void* d_workspace = nullptr;
    if (workspace_bytes > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));
    
    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, d_input, filter_desc, d_filter, conv_desc, conv_algo, d_workspace, workspace_bytes, &beta, conv_output_desc, d_conv_output));

    std::cout << "\n--- 2. Batch Normalization Step ---" << std::endl;

    cudnnTensorDescriptor_t bn_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bn_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bn_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv_out_C, 1, 1));
    
    float *d_bn_scale, *d_bn_bias;
    std::vector<float> h_bn_scale(conv_out_C, 1.0f); // Initialize scale to 1
    std::vector<float> h_bn_bias(conv_out_C, 0.0f);  // Initialize bias to 0
    CHECK_CUDA(cudaMalloc(&d_bn_scale, conv_out_C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bn_bias, conv_out_C * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_bn_scale, h_bn_scale.data(), conv_out_C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bn_bias, h_bn_bias.data(), conv_out_C * sizeof(float), cudaMemcpyHostToDevice));
    
    float *d_bn_running_mean, *d_bn_running_var, *d_bn_saved_mean, *d_bn_saved_inv_var;
    CHECK_CUDA(cudaMalloc(&d_bn_running_mean, conv_out_C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bn_running_var, conv_out_C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bn_saved_mean, conv_out_C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bn_saved_inv_var, conv_out_C * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_bn_running_mean, 0, conv_out_C * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_bn_running_var, 0, conv_out_C * sizeof(float)));

    float* d_bn_output;
    CHECK_CUDA(cudaMalloc(&d_bn_output, conv_out_N * conv_out_C * conv_out_H * conv_out_W * sizeof(float)));
    
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        cudnn_handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
        conv_output_desc, d_conv_output, conv_output_desc, d_bn_output, bn_desc,
        d_bn_scale, d_bn_bias, 1.0, d_bn_running_mean, d_bn_running_var, 1e-5,
        d_bn_saved_mean, d_bn_saved_inv_var));
    
    std::cout << "\n--- 3. ReLU Activation Step ---" << std::endl;
    cudnnActivationDescriptor_t activation_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    float* d_relu_output = d_bn_output; 
    CHECK_CUDNN(cudnnActivationForward(cudnn_handle, activation_desc, &alpha, conv_output_desc, d_bn_output, &beta, conv_output_desc, d_relu_output));

    std::cout << "\n--- 4. Max Pooling Step ---" << std::endl;
    cudnnPoolingDescriptor_t pooling_desc;
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pooling_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));

    int pool_out_N, pool_out_C, pool_out_H, pool_out_W;
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(pooling_desc, conv_output_desc, &pool_out_N, &pool_out_C, &pool_out_H, &pool_out_W));
    
    float* d_pool_output;
    CHECK_CUDA(cudaMalloc(&d_pool_output, pool_out_N * pool_out_C * pool_out_H * pool_out_W * sizeof(float)));
    
    CHECK_CUDNN(cudnnPoolingForward(cudnn_handle, pooling_desc, &alpha, conv_output_desc, d_relu_output, &beta, /* Note: need an output descriptor here */ conv_output_desc, d_pool_output));
    
    std::cout << "\nFull CNN layer forward pass complete." << std::endl;
    print_tensor("Final Output Tensor", d_pool_output, pool_out_N, pool_out_C, pool_out_H, pool_out_W);
    
    // --- Cleanup ---
    cudnnDestroy(cudnn_handle);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(conv_output_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyActivationDescriptor(activation_desc);
    cudnnDestroyPoolingDescriptor(pooling_desc);

    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_conv_output);
    cudaFree(d_bn_scale); cudaFree(d_bn_bias); cudaFree(d_bn_output);
    cudaFree(d_bn_running_mean); cudaFree(d_bn_running_var);
    cudaFree(d_bn_saved_mean); cudaFree(d_bn_saved_inv_var);
    cudaFree(d_pool_output);
    if (d_workspace) cudaFree(d_workspace);

    return 0;
}
