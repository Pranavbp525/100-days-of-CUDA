#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define SECTION_SIZE 1024  

__global__ void kogge_exclusive_scan_kernel(float *x, float *y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N && threadIdx.x !=0) {
        XY[threadIdx.x] = x[i];
    } else {
        XY[threadIdx.x] = 0.0f;
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp = 0;
        if (threadIdx.x >= stride) {
            temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            XY[threadIdx.x] = temp;
        }
    }

    if (i < N) {
        y[i] = XY[threadIdx.x];
    }
}

void kogge_exclusive_scan_host(float *x_h, float *y_h, unsigned int N) {
    if (N > SECTION_SIZE) {
        cout << "Input size N must be <= SECTION_SIZE (" << SECTION_SIZE << ")." << endl;
        return;
    }

    float *x_d, *y_d;
    cudaMalloc(&x_d, N * sizeof(float));
    cudaMalloc(&y_d, N * sizeof(float));
    cudaMemcpy(x_d, x_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(1);
    dim3 blockDim(SECTION_SIZE);
    kogge_exclusive_scan_kernel<<<gridDim, blockDim>>>(x_d, y_d, N);

    cudaMemcpy(y_h, y_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(y_d);
}

int main() {
    const unsigned int N = 1024;
    float *x_h = new float[N];
    float *y_h = new float[N];

    for (unsigned int i = 0; i < N; i++) {
        x_h[i] = 1.0f;
    }

    kogge_exclusive_scan_host(x_h, y_h, N);

    cout << "Exclusive Scan Result (first 10 elements):" << endl;
    for (unsigned int i = 0; i < 10 && i < N; i++) {
        cout << y_h[i] << " ";
    }
    cout << endl;

    delete[] x_h;
    delete[] y_h;
    return 0;
}