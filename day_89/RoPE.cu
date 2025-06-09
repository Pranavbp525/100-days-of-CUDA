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


__global__ void apply_rope_kernel(float* matrix, int batch_size, int seq_len, int embed_dim) {
    int pos = blockIdx.y;      
    int batch = blockIdx.z;    
    int pair_idx = threadIdx.x; 

    int dim_even = 2 * pair_idx;
    int dim_odd = dim_even + 1;

    if (batch < batch_size && pos < seq_len && dim_odd < embed_dim) {
        float freq = 1.0f / powf(10000.0f, (float)dim_even / (float)embed_dim);
        float angle = (float)pos * freq;
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);

        int base_idx = batch * seq_len * embed_dim + pos * embed_dim;
        int idx_even = base_idx + dim_even;
        int idx_odd = base_idx + dim_odd;

        float val_even = matrix[idx_even];
        float val_odd = matrix[idx_odd];

        matrix[idx_even] = val_even * cos_angle - val_odd * sin_angle;
        matrix[idx_odd] = val_even * sin_angle + val_odd * cos_angle;
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
    const int embed_dim = 512;

    std::cout << "--- Rotary Positional Encoding (RoPE) Application ---" << std::endl;
    std::cout << "Batch: " << batch_size << ", Seq Len: " << seq_len << ", Embed Dim: " << embed_dim << std::endl << std::endl;


    std::vector<float> h_query_matrix(batch_size * seq_len * embed_dim);
    for(size_t i = 0; i < h_query_matrix.size(); ++i) {
        h_query_matrix[i] = 1.0f;
    }

    float *d_query_matrix;
    CHECK_CUDA(cudaMalloc(&d_query_matrix, h_query_matrix.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_query_matrix, h_query_matrix.data(), h_query_matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Original Query Matrix (before RoPE)." << std::endl;
    print_matrix("Original Query Matrix", d_query_matrix, batch_size * seq_len, embed_dim);
    
    dim3 rope_blocks(1, seq_len, batch_size);
    dim3 rope_threads(embed_dim / 2); 
    apply_rope_kernel<<<rope_blocks, rope_threads>>>(d_query_matrix, batch_size, seq_len, embed_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << "Applied RoPE to the Query Matrix." << std::endl;
    print_matrix("Final Query Matrix (after RoPE)", d_query_matrix, batch_size * seq_len, embed_dim);
    

    std::vector<float> h_output(h_query_matrix.size());
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_query_matrix, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool pos0_unchanged = true;
    for(int i = 0; i < embed_dim; ++i) {
        if (std::fabs(h_output[i] - 1.0f) > 1e-6) {
            pos0_unchanged = false;
            break;
        }
    }
    std::cout << "Verification Check (Position 0 unchanged): " << (pos0_unchanged ? "PASSED" : "FAILED") << std::endl;

    bool pos1_changed = false;
    for(int i = 0; i < embed_dim; ++i) {
        if (std::fabs(h_output[embed_dim + i] - 1.0f) > 1e-6) {
            pos1_changed = true;
            break;
        }
    }
    std::cout << "Verification Check (Position 1 changed):   " << (pos1_changed ? "PASSED" : "FAILED") << std::endl;

    CHECK_CUDA(cudaFree(d_query_matrix));

    return 0;
}
