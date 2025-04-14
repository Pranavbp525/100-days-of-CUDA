#include <iostream>
using namespace std;

#define SECTION_SIZE 1024
#define THREADS_PER_BLOCK 256

__global__ void local_scan_kernel(float *X, float *Y, float *S, unsigned int N) {
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

    if (tid == THREADS_PER_BLOCK - 1) {
        S[blockIdx.x] = last_elements[tid];
    }
}

__global__ void scan_s_kernel(float *S, unsigned int num_blocks) {
    __shared__ float temp[SECTION_SIZE];
    unsigned int tid = threadIdx.x;

    if (tid < num_blocks) {
        temp[tid] = S[tid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride <= num_blocks; stride *= 2) {
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

    if (tid < num_blocks) {
        S[tid] = temp[tid];
    }
}

__global__ void add_prefix_kernel(float *Y, float *S, unsigned int N) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_id = blockIdx.x;

    if (block_id > 0 && idx < N) {
        Y[idx] += S[block_id - 1];
    }
}

void hierarchical_scan_host(float *X_h, float *Y_h, unsigned int N) {
    unsigned int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    float *X_d, *Y_d, *S_d;

    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));
    cudaMalloc(&S_d, num_blocks * sizeof(float));

    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim1(num_blocks);
    dim3 blockDim1(THREADS_PER_BLOCK);
    local_scan_kernel<<<gridDim1, blockDim1>>>(X_d, Y_d, S_d, N);

    dim3 gridDim2(1);
    dim3 blockDim2(SECTION_SIZE);
    scan_s_kernel<<<gridDim2, blockDim2>>>(S_d, num_blocks);

    dim3 gridDim3(num_blocks);
    dim3 blockDim3(SECTION_SIZE);
    add_prefix_kernel<<<gridDim3, blockDim3>>>(Y_d, S_d, N);

    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(S_d);
}

int main() {
    const unsigned int N = 4096;  
    float *X_h = new float[N];    
    float *Y_h = new float[N];    

    for (unsigned int i = 0; i < N; i++) {
        X_h[i] = 1.0f;  
    }


    hierarchical_scan_host(X_h, Y_h, N);

    
    cout << "Inclusive Scan Result (elements 1024 to 1123):" << endl;
    for (unsigned int i = 1024; i < 1124 && i < N; i++) {
        cout << Y_h[i] << " ";
    }
    cout << endl;

    
    delete[] X_h;
    delete[] Y_h;

    return 0;
}