#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define TILE_SIZE 2 

__global__ void tiledMatrixMulKernel(float* inp1, float* inp2, float* out, int size) {
    
    
    __shared__ float inp1_s[TILE_SIZE][TILE_SIZE];
    __shared__ float inp2_s[TILE_SIZE][TILE_SIZE];

    int i = threadIdx.y + blockIdx.y * TILE_SIZE;
    int j = threadIdx.x + blockIdx.x * TILE_SIZE;

    int num_phases = size/TILE_SIZE;

    float val = 0;

    for(int phase = 0; phase < num_phases; ++phase){

        int inp1_row_index = i;
        int inp1_col_index = phase*TILE_SIZE + threadIdx.x;

        inp1_s[threadIdx.y][threadIdx.x] = inp1[inp1_row_index*size + inp1_col_index];

        int inp2_row_index = phase*TILE_SIZE + threadIdx.y;
        int inp2_col_index = j;

        inp2_s[threadIdx.y][threadIdx.x] = inp2[inp2_row_index*size + inp2_col_index];

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; ++k){
            val += inp1_s[threadIdx.y][k]*inp2_s[k][threadIdx.x];
        }
        __syncthreads();

    }
    out[i*size + j] = val;


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