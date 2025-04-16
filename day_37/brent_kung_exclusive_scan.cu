#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define SECTION_SIZE 64

__global__ void brent_kung_exclusive_scan_kernel(float * X, float * Y, unsigned int N){
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        XY[threadIdx.x] = X[i];
    }
    if (i + blockDim.x < N) {
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE) {
            XY[index] += XY[index - stride];
        }
    }

    for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index + stride < SECTION_SIZE) {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();
    if (i < N) {
        if (i == 0) {
            Y[i] = 0;
        } else {
            Y[i] = XY[i - 1];
        }
    }
    if (i + blockDim.x < N) {
        Y[i + blockDim.x] = XY[i + blockDim.x - 1];
    }
}

void brent_kung_exclusive_scan_host(float *X_h, float *Y_h, unsigned int N) {
    if (N > SECTION_SIZE) {
        cout << "Input size N must be <= SECTION_SIZE (" << SECTION_SIZE << ")." << endl;
        return;
    }

    float *X_d, *Y_d;

    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));

    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(1);          
    dim3 blockDim(SECTION_SIZE / 2);  

    brent_kung_exclusive_scan_kernel<<<gridDim, blockDim>>>(X_d, Y_d, N);

    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Y_d);
}

int main() {
    const unsigned int N = 64;  
    float *X_h = new float[N];    
    float *Y_h = new float[N];    

    for (unsigned int i = 0; i < N; i++) {
        X_h[i] = 1.0f;  
    }

    brent_kung_exclusive_scan_host(X_h, Y_h, N);

    cout << "Exclusive Scan Result :" << endl;
    for (unsigned int i = 0; i < N; i++) {
        cout << Y_h[i] << " ";
    }
    cout << endl;

    delete[] X_h;
    delete[] Y_h;

    return 0;
}