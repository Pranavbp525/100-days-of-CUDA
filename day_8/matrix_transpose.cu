#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixTransposeKernel(float* inp, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        out[col * rows + row] = inp[row * cols + col];
    }
}

void matrixTransposeHost(float* inp_h, float* out_h, int rows, int cols) {
    int inp_size = rows * cols * sizeof(float);
    int out_size = rows * cols * sizeof(float);

    float *inp_d, *out_d;

    cudaMalloc((void**)&inp_d, inp_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(inp_d, inp_h, inp_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1); 
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, 1);

    matrixTransposeKernel<<<gridDim, blockDim>>>(inp_d, out_d, rows, cols);

    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    const int rows = 4;
    const int cols = 3;
    const int size = rows * cols;

    float* inp_h = new float[size];
    float* out_h = new float[size];

    
    for (int i = 0; i < size; i++) {
        inp_h[i] = static_cast<float>(i + 1); 
    }

    matrixTransposeHost(inp_h, out_h, rows, cols);

    
    cout << "Input Matrix (" << rows << "x" << cols << "):" << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << inp_h[i * cols + j] << " ";
        }
        cout << endl;
    }

    
    cout << "\nTransposed Matrix (" << cols << "x" << rows << "):" << endl;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            cout << out_h[i * rows + j] << " ";
        }
        cout << endl;
    }

    delete[] inp_h;
    delete[] out_h;

    return 0;
}