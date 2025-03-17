#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define TILE_SIZE 2

__global__ void tiledMatrixMulKernel(float* inp1, float* inp2, float* out, int dim_1, int dim_2, int dim_3) {
    __shared__ float inp1_s[TILE_SIZE][TILE_SIZE];
    __shared__ float inp2_s[TILE_SIZE][TILE_SIZE];

    int i = threadIdx.y + blockIdx.y * TILE_SIZE;
    int j = threadIdx.x + blockIdx.x * TILE_SIZE;

    float val = 0;

    int num_phases = ceil(float(dim_2) / TILE_SIZE);

    for (int phase = 0; phase < num_phases; ++phase) {
        int inp1_row_index = i;
        int inp1_col_index = phase * TILE_SIZE + threadIdx.x;

        if (inp1_row_index < dim_1 && inp1_col_index < dim_2) {
            inp1_s[threadIdx.y][threadIdx.x] = inp1[inp1_row_index * dim_2 + inp1_col_index];
        } else {
            inp1_s[threadIdx.y][threadIdx.x] = 0;
        }

        int inp2_row_index = phase * TILE_SIZE + threadIdx.y;
        int inp2_col_index = j;

        if (inp2_row_index < dim_2 && inp2_col_index < dim_3) {
            inp2_s[threadIdx.y][threadIdx.x] = inp2[inp2_row_index * dim_3 + inp2_col_index];
        } else {
            inp2_s[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            val += inp1_s[threadIdx.y][k] * inp2_s[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (i < dim_1 && j < dim_3) {
        out[i * dim_3 + j] = val;
    }
}

void tiledMatrixMulHost(float* inp1_h, float* inp2_h, float* out_h, int dim_1, int dim_2, int dim_3) {
    int inp1_size = dim_1 * dim_2 * sizeof(float);
    int inp2_size = dim_2 * dim_3 * sizeof(float);
    int out_size = dim_1 * dim_3 * sizeof(float);

    float *inp1_d, *inp2_d, *out_d;

    cudaMalloc((void**)&inp1_d, inp1_size);
    cudaMalloc((void**)&inp2_d, inp2_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(inp1_d, inp1_h, inp1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(inp2_d, inp2_h, inp2_size, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim(ceil(float(dim_3) / TILE_SIZE), ceil(float(dim_1) / TILE_SIZE), 1);

    tiledMatrixMulKernel<<<gridDim, blockDim>>>(inp1_d, inp2_d, out_d, dim_1, dim_2, dim_3);

    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);

    cudaFree(inp1_d);
    cudaFree(inp2_d);
    cudaFree(out_d);
}

int main() {
    const int dim_1 = 3; // Rows of inp1
    const int dim_2 = 4; // Columns of inp1, rows of inp2
    const int dim_3 = 2; // Columns of inp2

    const int inp1_size = dim_1 * dim_2;
    const int inp2_size = dim_2 * dim_3;
    const int out_size = dim_1 * dim_3;

    float* inp1_h = new float[inp1_size];
    float* inp2_h = new float[inp2_size];
    float* out_h = new float[out_size];

    // Initialize matrices
    for (int i = 0; i < inp1_size; i++) {
        inp1_h[i] = static_cast<float>(i + 1); // 1 to 12
    }
    for (int i = 0; i < inp2_size; i++) {
        inp2_h[i] = static_cast<float>(i + 1); // 1 to 8
    }

    // Perform multiplication
    tiledMatrixMulHost(inp1_h, inp2_h, out_h, dim_1, dim_2, dim_3);

    // Print results
    cout << "Matrix 1 (3x4):" << endl;
    for (int i = 0; i < dim_1; i++) {
        for (int j = 0; j < dim_2; j++) {
            cout << inp1_h[i * dim_2 + j] << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix 2 (4x2):" << endl;
    for (int i = 0; i < dim_2; i++) {
        for (int j = 0; j < dim_3; j++) {
            cout << inp2_h[i * dim_3 + j] << " ";
        }
        cout << endl;
    }

    cout << "\nResult (3x2):" << endl;
    for (int i = 0; i < dim_1; i++) {
        for (int j = 0; j < dim_3; j++) {
            cout << out_h[i * dim_3 + j] << " ";
        }
        cout << endl;
    }

    delete[] inp1_h;
    delete[] inp2_h;
    delete[] out_h;

    return 0;
}