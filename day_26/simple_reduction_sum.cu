#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void simpleReductionSumKernel(float *inp, float *out) {
    int i = 2 * threadIdx.x;
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            inp[i] += inp[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out = inp[0];
    }
}

void simpleReductionSumHost(float *inp_h, float *out_h, int length) {
    if (length % 2 != 0 || length > 2048) {
        cout << "Input length must be even and <= 2048 for this single-block reduction." << endl;
        return;
    }
    int block_size = length / 2;
    if ((block_size & (block_size - 1)) != 0) {
        cout << "Block size must be a power of 2 for efficient reduction." << endl;
        return;
    }

    float *inp_d, *out_d;
    cudaMalloc(&inp_d, length * sizeof(float));
    cudaMalloc(&out_d, sizeof(float));

    cudaMemcpy(inp_d, inp_h, length * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(1);
    dim3 blockDim(block_size);

    simpleReductionSumKernel<<<gridDim, blockDim>>>(inp_d, out_d);

    cudaMemcpy(out_h, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    const int length = 1024;
    float *inp_h = new float[length];
    for (int i = 0; i < length; i++) {
        inp_h[i] = 1.0f;
    }
    float out_h = 0.0f;

    simpleReductionSumHost(inp_h, &out_h, length);

    cout << "Sum: " << out_h << endl;

    delete[] inp_h;
    return 0;
}