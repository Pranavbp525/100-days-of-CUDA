#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define BLOCKDIM 512
#define COARSEFACTOR 2

__global__ void segmentedSumReductionKernel(float *input, float *out) {
    __shared__ float input_s[BLOCKDIM];
    unsigned int segment = COARSEFACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = input[i];
    for (unsigned int tile = 1; tile < 2 * COARSEFACTOR; ++tile) {
        sum += input[i + tile * BLOCKDIM];
    }
    input_s[t] = sum;
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(out, input_s[0]);
    }
}

void segmentedSumReductionHost(float *inp_h, float *out_h, int length) {
    int block_size = BLOCKDIM; 
    int elements_per_block = 2 * COARSEFACTOR * block_size;  
    int grid_size = (length + elements_per_block - 1) / elements_per_block;

    if ((block_size & (block_size - 1)) != 0) {
        cout << "Block size must be a power of 2." << endl;
        return;
    }
    if (length % elements_per_block != 0) {
        cout << "Input length should be a multiple of " << elements_per_block << " for this implementation." << endl;
    }

    float *inp_d, *out_d;
    cudaMalloc(&inp_d, length * sizeof(float));
    cudaMalloc(&out_d, sizeof(float));

    cudaMemcpy(inp_d, inp_h, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, sizeof(float));

    segmentedSumReductionKernel<<<grid_size, block_size>>>(inp_d, out_d);

    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    const int block_size = BLOCKDIM; 
    const int grid_size = 4;
    const int length = COARSEFACTOR * 2 * block_size * grid_size;  

    float *inp_h = new float[length];
    float out_h = 0.0f;

    for (int i = 0; i < length; i++) {
        inp_h[i] = 1.0f;
    }

    segmentedSumReductionHost(inp_h, &out_h, length);

    cout << "Sum of " << length << " elements: " << out_h << endl;

    delete[] inp_h;
    return 0;
}