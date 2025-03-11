#include <iostream>
#include <cmath>

__global__ void vectorAddKernel(float* A, float* B, float* C, int n){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<n){

        C[i] = A[i] + B[i];

    }

}

void vector_add(float *A_h, float *B_h, float *C_h, int n) {

    int size = n * sizeof(float);

    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    int block_size = 256.0;
    int grid_size = (n + block_size - 1) / block_size;

    vectorAddKernel<<grid_size, block_size>>(A_d, B_d, C_d, n);

    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d); 


}

int main(){

    int n = 10000;
    int size = n * sizeof(float);

    float* A_h = new float[n];
    float* B_h = new float[n];
    float* C_h = new float[n]; 
    
    


}