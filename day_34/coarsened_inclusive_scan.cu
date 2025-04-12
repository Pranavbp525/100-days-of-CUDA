#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define THREADS_PER_BLOCK 64
#define SECTION_SIZE 1024  

__global__ void coarsened_scan_kernel(float *X, float *Y, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];          
    __shared__ float last_elements[THREADS_PER_BLOCK]; 

    unsigned int tid = threadIdx.x;
    unsigned int block_offset = blockIdx.x * SECTION_SIZE;
    unsigned int subsection_size = SECTION_SIZE / THREADS_PER_BLOCK;

    
    float sum = 0.0f;
    for (unsigned int i = 0; i < subsection_size; ++i) {
        unsigned int global_idx = block_offset + tid * subsection_size + i;
        if (global_idx < N) {
            XY[tid * subsection_size + i] = X[global_idx]; 
            sum += XY[tid * subsection_size + i];
            XY[tid * subsection_size + i] = sum; 
        } else {
            XY[tid * subsection_size + i] = 0.0f; 
        }
    }
    last_elements[tid] = sum; 
    __syncthreads();

    for (unsigned int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = last_elements[tid] + last_elements[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            last_elements[tid] = temp;
        }
        __syncthreads();
    }

    if (tid > 0) {
        float prefix_sum = last_elements[tid - 1];
        for (unsigned int i = 0; i < subsection_size; ++i) {
            unsigned int idx = tid * subsection_size + i;
            if (block_offset + idx < N) {
                Y[block_offset + idx] = XY[idx] + prefix_sum;
            }
        }
    } else {
        
        for (unsigned int i = 0; i < subsection_size; ++i) {
            unsigned int idx = tid * subsection_size + i;
            if (block_offset + idx < N) {
                Y[block_offset + idx] = XY[idx];
            }
        }
    }
}

void coarsened_scan_host(float *X_h, float *Y_h, unsigned int N) {
    unsigned int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    float *X_d, *Y_d;

    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));

    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(num_blocks);
    dim3 blockDim(THREADS_PER_BLOCK);

    coarsened_scan_kernel<<<gridDim, blockDim>>>(X_d, Y_d, N);

    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Y_d);
}

int main() {
    const unsigned int N = 4096;  
    float *X_h = new float[N];    
    float *Y_h = new float[N];    

    for (unsigned int i = 0; i < N; i++) {
        X_h[i] = 1.0f;  
    }

    coarsened_scan_host(X_h, Y_h, N);

    cout << "Inclusive Scan Result (first 10 elements):" << endl;
    for (unsigned int i = 0; i < 10 && i < N; i++) {
        cout << Y_h[i] << " ";
    }
    cout << endl;

    delete[] X_h;
    delete[] Y_h;

    return 0;
}


