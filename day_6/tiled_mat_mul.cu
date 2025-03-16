#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define TILE_SIZE 16 

__global__ void tiledMatrixMulKernel(float* inp1, float* inp2, float* out, int size) {
    
    __shared__ float tile1[TILE_SIZE][TILE_SIZE];
    __shared__ float tile2[TILE_SIZE][TILE_SIZE];

    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float val = 0;

    
    for (int t = 0; t < (size + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        if (row < size && (t * TILE_SIZE + threadIdx.x) < size) {
            tile1[threadIdx.y][threadIdx.x] = inp1[row * size + t * TILE_SIZE + threadIdx.x];
        } else {
            tile1[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < size && (t * TILE_SIZE + threadIdx.y) < size) {
            tile2[threadIdx.y][threadIdx.x] = inp2[(t * TILE_SIZE + threadIdx.y) * size + col];
        } else {
            tile2[threadIdx.y][threadIdx.x] = 0;
        }

        
        __syncthreads();

        
        for (int k = 0; k < TILE_SIZE; ++k) {
            val += tile1[threadIdx.y][k] * tile2[k][threadIdx.x];
        }

        
        __syncthreads();
    }

    
    if (row < size && col < size) {
        out[row * size + col] = val;
    }
}

void tiledMatrixMulHost(float* inp1_h, float* inp2_h, float* out_h, int size) {
    int total_size = size * size * sizeof(float);

    float *inp1_d, *inp2_d, *out_d;

    cudaMalloc((void**)&inp1_d, total_size);
    cudaMalloc((void**)&inp2_d, total_size);
    cudaMalloc((void**)&out_d, total_size);

    cudaMemcpy(inp1_d, inp1_h, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(inp2_d, inp2_h, total_size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1); 
    dim3 gridDim((size + TILE_SIZE - 1) / TILE_SIZE, (size + TILE_SIZE - 1) / TILE_SIZE, 1);

    tiledMatrixMulKernel<<<gridDim, blockDim>>>(inp1_d, inp2_d, out_d, size);

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

    
    tiledMatrixMulHost(inp1_h, inp2_h, out_h, size);

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