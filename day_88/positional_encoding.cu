#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}


__global__ void generate_positional_encoding_kernel(float* pe_matrix, int seq_len, int embed_dim) {
    int pos = blockIdx.x;  
    int i = threadIdx.x;   

    if (pos < seq_len && i < embed_dim) {
        float div_term = powf(10000.0f, (float)(2 * (i / 2)) / (float)embed_dim);
        float angle = (float)pos / div_term;
        
        int idx = pos * embed_dim + i;
        
        if (i % 2 == 0) {
            pe_matrix[idx] = sinf(angle);
        } else {
            pe_matrix[idx] = cosf(angle);
        }
    }
}


__global__ void add_positional_encoding_kernel(float* embeddings, const float* pe_matrix, int batch_size, int seq_len, int embed_dim) {
    int batch = blockIdx.z;
    int pos = blockIdx.y;
    int dim = threadIdx.x;

    if (batch < batch_size && pos < seq_len && dim < embed_dim) {
        int embedding_idx = batch * seq_len * embed_dim + pos * embed_dim + dim;
        int pe_idx = pos * embed_dim + dim;
        
        embeddings[embedding_idx] += pe_matrix[pe_idx];
    }
}


void print_matrix(const std::string& name, const float* m, int rows, int cols) {
    std::cout << "--- " << name << " --- (" << rows << "x" << cols << ")\n";
    std::vector<float> h_m(rows * cols);
    CHECK_CUDA(cudaMemcpy(h_m.data(), m, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < std::min(8, rows); ++i) {
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
    const int embed_dim = 512;

    std::cout << "--- Positional Encoding Generation & Application ---" << std::endl;
    std::cout << "Batch: " << batch_size << ", Seq Len: " << seq_len << ", Embed Dim: " << embed_dim << std::endl << std::endl;

    
    std::vector<float> h_input_embeddings(batch_size * seq_len * embed_dim, 0.0f);

    float *d_input_embeddings, *d_pe_matrix, *d_output_embeddings;

    CHECK_CUDA(cudaMalloc(&d_input_embeddings, h_input_embeddings.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pe_matrix, seq_len * embed_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_embeddings, h_input_embeddings.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input_embeddings, h_input_embeddings.data(), h_input_embeddings.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output_embeddings, d_input_embeddings, h_input_embeddings.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    
    dim3 pe_blocks(seq_len);
    dim3 pe_threads(embed_dim);
    generate_positional_encoding_kernel<<<pe_blocks, pe_threads>>>(d_pe_matrix, seq_len, embed_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << "Generated Positional Encoding Matrix." << std::endl;
    print_matrix("Positional Encoding Matrix", d_pe_matrix, seq_len, embed_dim);

    dim3 add_blocks(1, seq_len, batch_size);
    dim3 add_threads(embed_dim);
    add_positional_encoding_kernel<<<add_blocks, add_threads>>>(d_output_embeddings, d_pe_matrix, batch_size, seq_len, embed_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Added Positional Encodings to Input Embeddings." << std::endl;
    print_matrix("Final Output Embeddings (Input + PE)", d_output_embeddings, batch_size * seq_len, embed_dim);
    

    std::vector<float> h_output(h_input_embeddings.size());
    std::vector<float> h_pe(seq_len * embed_dim);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output_embeddings, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_pe.data(), d_pe_matrix, h_pe.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool match = true;
    for(int i = 0; i < embed_dim; ++i) {
        if (std::fabs(h_output[i] - h_pe[i]) > 1e-6) {
            match = false;
            break;
        }
    }
    std::cout << "Verification Check: " << (match ? "PASSED" : "FAILED") << std::endl;


    CHECK_CUDA(cudaFree(d_input_embeddings));
    CHECK_CUDA(cudaFree(d_pe_matrix));
    CHECK_CUDA(cudaFree(d_output_embeddings));

    return 0;
}
