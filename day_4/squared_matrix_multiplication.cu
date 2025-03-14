#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixMulKernel(float* inp1, float* inp2, float* out, int size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < size && col < size) {
        float val = 0;
        for (int k = 0; k < size; ++k) {
            val += inp1[row * size + k] * inp2[k * size + col];
        }
        out[row * size + col] = val;
    }
}

void matrixMulHost(float* inp1_h, float* inp2_h, float* out_h, int size) {
    int total_size = size * size * sizeof(float);

    float *inp1_d, *inp2_d, *out_d;

    cudaMalloc((void**)&inp1_d, total_size);
    cudaMalloc((void**)&inp2_d, total_size);
    cudaMalloc((void**)&out_d, total_size);

    cudaMemcpy(inp1_d, inp1_h, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(inp2_d, inp2_h, total_size, cudaMemcpyHostToDevice);

    dim3 blockDim(2, 2, 1);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y, 1);

    matrixMulKernel<<<gridDim, blockDim>>>(inp1_d, inp2_d, out_d, size);

    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, total_size, cudaMemcpyDeviceToHost);

    cudaFree(inp1_d);
    cudaFree(inp2_d);
    cudaFree(out_d);
}

int main() {
    const int size = 4;
    const int total_size = size * size;

    
    float* inp1_h = new float[total_size];
    float* inp2_h = new float[total_size];
    float* out_h = new float[total_size];

    
    for (int i = 0; i < total_size; i++) {
        inp1_h[i] = static_cast<float>(i + 1); 
        inp2_h[i] = static_cast<float>(i + 1); 
    }

    
    matrixMulHost(inp1_h, inp2_h, out_h, size);

    
    cout << "Matrix 1:" << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << inp1_h[i * size + j] << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix 2:" << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << inp2_h[i * size + j] << " ";
        }
        cout << endl;
    }

    cout << "\nResult:" << endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << out_h[i * size + j] << " ";
        }
        cout << endl;
    }

    
    delete[] inp1_h;
    delete[] inp2_h;
    delete[] out_h;

    return 0;
}
