#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define THREADS_PER_BLOCK 64
#define SECTION_SIZE 1024  

__global__ void coarsened_exclusive_scan_kernel(float *X, float *Y, float *S, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];          
    __shared__ float last_elements[THREADS_PER_BLOCK]; 

    unsigned int tid = threadIdx.x;
    unsigned int block_offset = blockIdx.x * SECTION_SIZE;
    unsigned int subsection_size = SECTION_SIZE / THREADS_PER_BLOCK;

    float prev = 0.0f;
    for (unsigned int i = 0; i < subsection_size; ++i) {
        unsigned int global_idx = block_offset + tid * subsection_size + i;
        if (global_idx < N) {
            float temp = X[global_idx];
            XY[tid * subsection_size + i] = prev;
            prev += temp;
        } else {
            XY[tid * subsection_size + i] = prev;
        }
    }
    last_elements[tid] = prev;  // Total sum of the subsection
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

    if (tid == THREADS_PER_BLOCK - 1) {
        S[blockIdx.x] = last_elements[tid];
    }
}

__global__ void block_sum_exclusive_scan_kernel(float *S, float *prefix, unsigned int num_blocks) {
    __shared__ float temp[SECTION_SIZE];
    unsigned int tid = threadIdx.x;

    if (tid < num_blocks) {
        temp[tid] = S[tid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride < num_blocks; stride *= 2) {
        float val = 0.0f;
        if (tid >= stride) {
            val = temp[tid] + temp[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            temp[tid] = val;
        }
        __syncthreads();
    }

    if (tid == 0) {
        prefix[0] = 0.0f;
    } else if (tid < num_blocks) {
        prefix[tid] = temp[tid - 1];
    }
}

__global__ void add_prefix_kernel(float *Y, float *prefix, unsigned int N, unsigned int section_size) {
    unsigned int b = blockIdx.x;
    float prefix_val = (b > 0) ? prefix[b] : 0.0f;
    unsigned int block_offset = b * section_size;
    unsigned int tid = threadIdx.x;
    while (tid < section_size) {
        unsigned int idx = block_offset + tid;
        if (idx < N) {
            Y[idx] += prefix_val;
        }
        tid += blockDim.x;
    }
}

void coarsened_exclusive_scan_host(float *X_h, float *Y_h, unsigned int N) {
    unsigned int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    float *X_d, *Y_d, *S_d, *prefix_d;

    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));
    cudaMalloc(&S_d, num_blocks * sizeof(float));
    cudaMalloc(&prefix_d, num_blocks * sizeof(float));

    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(num_blocks);
    dim3 blockDim(THREADS_PER_BLOCK);
    coarsened_exclusive_scan_kernel<<<gridDim, blockDim>>>(X_d, Y_d, S_d, N);

    dim3 gridDim2(1);
    dim3 blockDim2(SECTION_SIZE);
    block_sum_exclusive_scan_kernel<<<gridDim2, blockDim2>>>(S_d, prefix_d, num_blocks);

    dim3 gridDim3(num_blocks);
    dim3 blockDim3(256);
    add_prefix_kernel<<<gridDim3, blockDim3>>>(Y_d, prefix_d, N, SECTION_SIZE);

    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(S_d);
    cudaFree(prefix_d);
}

int main() {
    const unsigned int N = 4096;  
    float *X_h = new float[N];    
    float *Y_h = new float[N];    

    for (unsigned int i = 0; i < N; i++) {
        X_h[i] = 1.0f;  
    }

    coarsened_exclusive_scan_host(X_h, Y_h, N);

    cout << "Exclusive Scan Result (first 10 elements):" << endl;
    for (unsigned int i = 0; i < 10 && i < N; i++) {
        cout << Y_h[i] << " ";
    }
    cout << endl;

    cout << "Element at index 1024: " << Y_h[1024] << endl; 
    cout << "Element at index 2048: " << Y_h[2048] << endl; 

    delete[] X_h;
    delete[] Y_h;

    return 0;
}