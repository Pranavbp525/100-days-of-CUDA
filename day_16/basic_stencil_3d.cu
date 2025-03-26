#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Declare constant coefficients in device memory
__constant__ float c0;
__constant__ float c1;
__constant__ float c2;
__constant__ float c3;
__constant__ float c4;
__constant__ float c5;
__constant__ float c6;


__global__ void basic_stencil_3d(float *inp, float *out, int side) {
    int i = threadIdx.z + blockIdx.z * blockDim.z;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= 1 && i < side - 1 && j >= 1 && j < side - 1 && k >= 1 && k < side - 1) {
        out[i * side * side + j * side + k] = 
            c0 * inp[i * side * side + j * side + k] +
            c1 * inp[i * side * side + j * side + (k - 1)] +
            c2 * inp[i * side * side + j * side + (k + 1)] +
            c3 * inp[i * side * side + (j - 1) * side + k] +
            c4 * inp[i * side * side + (j + 1) * side + k] +
            c5 * inp[(i - 1) * side * side + j * side + k] +
            c6 * inp[(i + 1) * side * side + j * side + k];
    }
}


void basic_stencil_3d_host(float *inp_h, float *out_h, int side) {
    int size = side * side * side;
    int bytes = size * sizeof(float);

    float *inp_d, *out_d;
    cudaMalloc(&inp_d, bytes);
    cudaMalloc(&out_d, bytes);

    cudaMemcpy(inp_d, inp_h, bytes, cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, bytes);

    dim3 blockDim(4, 4, 4);
    dim3 gridDim((side + blockDim.x - 1) / blockDim.x,
                 (side + blockDim.y - 1) / blockDim.y,
                 (side + blockDim.z - 1) / blockDim.z);

    basic_stencil_3d<<<gridDim, blockDim>>>(inp_d, out_d, side);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "Kernel launch failed: " << cudaGetErrorString(err) << endl;
        exit(1);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    const int side = 16;  
    int size = side * side * side;

    
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

    
    basic_stencil_3d_host(inp_h, out_h, side);

    
    cout << "Sample output (interior points: i=1, j=1, k=1 to 5):" << endl;
    for (int k = 1; k <= 5; k++) {
        int index = 1 * side * side + 1 * side + k;
        cout << out_h[index] << " ";
    }
    cout << endl;

    
    delete[] inp_h;
    delete[] out_h;

    return 0;
}