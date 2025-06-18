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
    const int N = 4;      
    const int C = 3;      
    const int H = 256;    
    const int W = 256;    

    const int K = 16;     
    const int R = 3;      
    const int S = 3;      

    std::cout << "--- 'Hello, cuDNN': 2D Convolution Forward Pass ---" << std::endl;
    std::cout << "Input: " << N << "x" << C << "x" << H << "x" << W 
              << " | Filter: " << K << "x" << C << "x" << R << "x" << S << std::endl;

    
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

    
    cudnnTensorDescriptor_t input_desc, output_desc, filter_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

    
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    int out_N, out_C, out_H, out_W;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &out_N, &out_C, &out_H, &out_W));
    
    float* d_output;
    CHECK_CUDA(cudaMalloc(&d_output, out_N * out_C * out_H * out_W * sizeof(float)));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_N, out_C, out_H, out_W));
    
    cudnnConvolutionFwdAlgoPerf_t perf_results[1];
    int returned_algo_count;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, 1, &returned_algo_count, perf_results));
    cudnnConvolutionFwdAlgo_t conv_algo = perf_results[0].algo;
    
    std::cout << "\ncuDNN chose algorithm: " << conv_algo << std::endl;

    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, conv_algo, &workspace_bytes));
    
    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));
    }
    std::cout << "Workspace size: " << workspace_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

    const float alpha = 1.0f, beta = 0.0f; 
    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, d_input, filter_desc, d_filter, conv_desc, conv_algo, d_workspace, workspace_bytes, &beta, output_desc, d_output));

    std::cout << "\nConvolution forward pass complete." << std::endl;
    print_tensor("Output Tensor", d_output, out_N, out_C, out_H, out_W);

    
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    if (d_workspace) {
        CHECK_CUDA(cudaFree(d_workspace));
    }

    return 0;
}
