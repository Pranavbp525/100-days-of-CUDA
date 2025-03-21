#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define FILTER_RADIUS 2
#define FULL_FILTER_SIZE (2 * FILTER_RADIUS + 1) 


__constant__ float F[FULL_FILTER_SIZE * FULL_FILTER_SIZE];

__global__ void convolution_2D_const_mem_kernel(float *N, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    int out_width = width - (2 * r + 1) + 1;
    int out_height = height - (2 * r + 1) + 1;

    if (outRow < out_height && outCol < out_width) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
                }
            }
        }
        P[outRow * out_width + outCol] = Pvalue;
    }
}

void convolution_2d_constant_host(float* N_h, float* filter_h, float* P_h, int r, int width, int height) {
    int input_size = width * height * sizeof(float);
    int output_size = (width - (2 * r + 1) + 1) * (height - (2 * r + 1) + 1) * sizeof(float);
    int filter_size_bytes = (2 * r + 1) * (2 * r + 1) * sizeof(float);

    float *N_d, *P_d;

    cudaMalloc((void**)&N_d, input_size);
    cudaMalloc((void**)&P_d, output_size);

    cudaMemcpy(N_d, N_h, input_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, filter_h, filter_size_bytes);

    dim3 blockDim(4, 4, 1);
    dim3 gridDim((width - (2 * r + 1) + 1 + blockDim.x - 1) / blockDim.x, 
                 (height - (2 * r + 1) + 1 + blockDim.y - 1) / blockDim.y, 1);

    convolution_2D_const_mem_kernel<<<gridDim, blockDim>>>(N_d, P_d, r, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(P_h, P_d, output_size, cudaMemcpyDeviceToHost);

    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    const int width = 16;
    const int height = 16;
    const int r = 2; 
    const int full_filter_size = 2 * r + 1;
    const int out_width = width - full_filter_size + 1;  
    const int out_height = height - full_filter_size + 1; 

    const int input_size = width * height;
    const int filter_size_total = full_filter_size * full_filter_size;
    const int output_size = out_width * out_height;

    float* N_h = new float[input_size];
    float* filter_h = new float[filter_size_total];
    float* P_h = new float[output_size];

    
    for (int i = 0; i < input_size; i++) {
        N_h[i] = static_cast<float>(i + 1);
    }

    
    for (int i = 0; i < filter_size_total; i++) {
        filter_h[i] = 1.0f / 25.0f;
    }

    convolution_2d_constant_host(N_h, filter_h, P_h, r, width, height);

    
    cout << "Input Matrix (16x16):" << endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << N_h[i * width + j] << " ";
        }
        cout << endl;
    }

    
    cout << "\nFilter (5x5):" << endl;
    for (int i = 0; i < full_filter_size; i++) {
        for (int j = 0; j < full_filter_size; j++) {
            cout << filter_h[i * full_filter_size + j] << " ";
        }
        cout << endl;
    }

    
    cout << "\nOutput Matrix (12x12):" << endl;
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            cout << P_h[i * out_width + j] << " ";
        }
        cout << endl;
    }

    delete[] N_h;
    delete[] filter_h;
    delete[] P_h;

    return 0;
}