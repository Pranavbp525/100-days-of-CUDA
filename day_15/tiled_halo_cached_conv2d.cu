#include <iostream>
#include <cuda_runtime.h>
using namespace std;


#define TILE_DIM 32
#define FILTER_RADIUS 1 
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)


__constant__ float F[FILTER_SIZE][FILTER_SIZE];


__global__ void convolution_2D_kernel(float *N, float *P, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float Ns[TILE_DIM][TILE_DIM];
    if (row < height && col < width) {
        Ns[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        Ns[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                if (threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
                    threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
                    threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
                    threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM) {
                    Pvalue += F[fRow][fCol] * Ns[threadIdx.y - FILTER_RADIUS + fRow][threadIdx.x - FILTER_RADIUS + fCol];
                }
            }
        }
        P[row * width + col] = Pvalue;
    } else {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                if (row - FILTER_RADIUS + fRow >= 0 &&
                    row - FILTER_RADIUS + fRow < height &&
                    col - FILTER_RADIUS + fCol >= 0 &&
                    col - FILTER_RADIUS + fCol < width) {
                    Pvalue += F[fRow][fCol] * N[(row - FILTER_RADIUS + fRow) * width + (col - FILTER_RADIUS + fCol)];
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

int main() {
    
    const int width = 64;
    const int height = 64;

    
    float *h_N = new float[width * height];
    float *h_P = new float[width * height];
    float h_F[FILTER_SIZE][FILTER_SIZE];

    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_N[i * width + j] = static_cast<float>(rand() % 10);  
        }
    }
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            h_F[i][j] = 1.0f / (FILTER_SIZE * FILTER_SIZE);  
        }
    }

    float *d_N, *d_P;
    cudaMalloc(&d_N, width * height * sizeof(float));
    cudaMalloc(&d_P, width * height * sizeof(float));

    cudaMemcpy(d_N, h_N, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_F, FILTER_SIZE * FILTER_SIZE * sizeof(float));  

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    convolution_2D_kernel<<<gridDim, blockDim>>>(d_N, d_P, width, height);

    

    cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Sample output (first 5x5 block):" << endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << h_P[i * width + j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_N);
    cudaFree(d_P);
    delete[] h_N;
    delete[] h_P;

    return 0;
}