#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define OUT_TILE_DIM 4
#define IN_TILE_DIM 6

__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_kernel(float *inp, float *out, int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x;

    float in_prev;
    float in_curr;
    float in_next;
    __shared__ float in_curr_s[IN_TILE_DIM][IN_TILE_DIM];
    

    
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_prev = inp[(iStart - 1) * N * N + j * N + k];
    }

    
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
         in_curr = inp[iStart * N * N + j * N + k];
         in_curr_s[threadIdx.y][threadIdx.x] = in_curr;
    }

    
    for (int tile_i = 0; tile_i < OUT_TILE_DIM; tile_i++) {
        int i = iStart + tile_i;

        
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            in_next = inp[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();

        
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1 &&
                threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] =
                    c0 * in_curr +
                    c1 * in_curr_s[threadIdx.y][threadIdx.x - 1] +
                    c2 * in_curr_s[threadIdx.y][threadIdx.x + 1] +
                    c3 * in_curr_s[threadIdx.y - 1][threadIdx.x] +
                    c4 * in_curr_s[threadIdx.y + 1][threadIdx.x] +
                    c5 * in_prev +
                    c6 * in_next;
            }
        }
        __syncthreads();

        
        if (tile_i < OUT_TILE_DIM - 1) {
            in_prev = in_curr;
            in_curr = in_next;
            in_curr_s[threadIdx.y][threadIdx.x] = in_curr;
        }
    }
}


void stencil_host(float *inp_h, float *out_h, int N) {
    int size = N * N * N;
    int bytes = size * sizeof(float);

    
    float *inp_d, *out_d;
    cudaMalloc(&inp_d, bytes);
    cudaMalloc(&out_d, bytes);
  
    cudaMemcpy(inp_d, inp_h, bytes, cudaMemcpyHostToDevice);

    cudaMemset(out_d, 0, bytes);
 
    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM, 1);

    dim3 gridDim((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    
    stencil_kernel<<<gridDim, blockDim>>>(inp_d, out_d, N);

    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}


int main() {
    const int N = 16; 
    int size = N * N * N;

    float *inp_h = new float[size];
    float *out_h = new float[size];

    for (int i = 0; i < size; i++) {
        inp_h[i] = static_cast<float>(i);
    }

    float h_c0 = 1.0f; 
    float h_c1 = 0.5f;
    float h_c2 = 0.5f; 
    float h_c3 = 0.5f; 
    float h_c4 = 0.5f; 
    float h_c5 = 0.5f; 
    float h_c6 = 0.5f; 

    cudaMemcpyToSymbol(c0, &h_c0, sizeof(float));
    cudaMemcpyToSymbol(c1, &h_c1, sizeof(float));
    cudaMemcpyToSymbol(c2, &h_c2, sizeof(float));
    cudaMemcpyToSymbol(c3, &h_c3, sizeof(float));
    cudaMemcpyToSymbol(c4, &h_c4, sizeof(float));
    cudaMemcpyToSymbol(c5, &h_c5, sizeof(float));
    cudaMemcpyToSymbol(c6, &h_c6, sizeof(float));

    stencil_host(inp_h, out_h, N);

    cout << "Sample output (interior points: i=1, j=1, k=1 to 5):" << endl;
    for (int k = 1; k <= 5; k++) {
        int i = 1, j = 1;
        int index = i * N * N + j * N + k;
        cout << out_h[index] << " ";
    }
    cout << endl;

    delete[] inp_h;
    delete[] out_h;

    return 0;
}