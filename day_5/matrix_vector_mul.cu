#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixVecMulKernel(float* mat, float* vec, float* out, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float val = 0;
        for (int i = 0; i < num_rows; ++i) {
            val += mat[row * num_rows + i] * vec[i]; 
        }
        out[row] = val;
    }
}

void matrixVecMulHost(float* mat_h, float* vec_h, float* out_h, int num_rows) {
    int matrix_size = num_rows * num_rows * sizeof(float);
    int vector_size = num_rows * sizeof(float);

    float *mat_d, *vec_d, *out_d;

    cudaMalloc((void**)&mat_d, matrix_size);
    cudaMalloc((void**)&vec_d, vector_size);
    cudaMalloc((void**)&out_d, vector_size);

    cudaMemcpy(mat_d, mat_h, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(vec_d, vec_h, vector_size, cudaMemcpyHostToDevice);

    dim3 blockDim(256, 1, 1);
    dim3 gridDim((num_rows + blockDim.x - 1) / blockDim.x, 1, 1);

    matrixVecMulKernel<<<gridDim, blockDim>>>(mat_d, vec_d, out_d, num_rows);

    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, vector_size, cudaMemcpyDeviceToHost);

    cudaFree(mat_d);
    cudaFree(vec_d);
    cudaFree(out_d);
}

int main() {
    const int num_rows = 4;
    const int matrix_size = num_rows * num_rows;
    const int vector_size = num_rows;

    
    float* mat_h = new float[matrix_size];
    float* vec_h = new float[vector_size];
    float* out_h = new float[vector_size];

    
    for (int i = 0; i < matrix_size; i++) {
        mat_h[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < vector_size; i++) {
        vec_h[i] = static_cast<float>(i + 1);
    }

    
    matrixVecMulHost(mat_h, vec_h, out_h, num_rows);

    
    cout << "Matrix:" << endl;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_rows; j++) {
            cout << mat_h[i * num_rows + j] << " ";
        }
        cout << endl;
    }

    cout << "\nVector:" << endl;
    for (int i = 0; i < num_rows; i++) {
        cout << vec_h[i] << " ";
    }
    cout << endl;

    cout << "\nResult:" << endl;
    for (int i = 0; i < num_rows; i++) {
        cout << out_h[i] << " ";
    }
    cout << endl;

    
    delete[] mat_h;
    delete[] vec_h;
    delete[] out_h;

    return 0;
}