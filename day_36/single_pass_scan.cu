#include <cuda_runtime.h>

#define SECTION_SIZE 256
#define THREADS_PER_BLOCK 64

__global__ void local_scan_kernel(float *X, float *Y, float *S, int N) {
    __shared__ float XY[SECTION_SIZE];
    __shared__ float last_elements[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SECTION_SIZE;
    int subsection_size = SECTION_SIZE / THREADS_PER_BLOCK;
    
    // Phase 1: Each thread performs a sequential scan on its subsection
    float sum = 0.0f;
    for (int i = 0; i < subsection_size; i++) {
        int global_idx = block_offset + tid * subsection_size + i;
        if (global_idx < N) {
            XY[tid * subsection_size + i] = X[global_idx];
            sum += XY[tid * subsection_size + i];
            XY[tid * subsection_size + i] = sum;
        }
    }
    last_elements[tid] = sum;
    __syncthreads();
    
    // Phase 2: Kogge-Stone scan on last_elements
    for (int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) {
        float temp;
        if (tid >= stride) {
            temp = last_elements[tid] + last_elements[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            last_elements[tid] = temp;
        }
        __syncthreads();
    }
    
    // Set block sum
    if (tid == THREADS_PER_BLOCK - 1) {
        S[blockIdx.x] = last_elements[tid]; // Corrected: Use only the last element
    }
    
    // Phase 3: Write results to global memory
    for (int i = 0; i < subsection_size; i++) {
        int idx = tid * subsection_size + i;
        int global_idx = block_offset + idx;
        if (global_idx < N) {
            if (tid == 0) {
                Y[global_idx] = XY[idx];
            } else {
                Y[global_idx] = XY[idx] + last_elements[tid - 1];
            }
        }
    }
}

__global__ void scan_s_kernel(float *S, int num_blocks) {
    __shared__ float temp[SECTION_SIZE];
    int tid = threadIdx.x;
    if (tid < num_blocks) {
        temp[tid] = S[tid];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();
    
    for (int stride = 1; stride < SECTION_SIZE; stride *= 2) {
        float val;
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

__global__ void add_prefix_kernel(float *Y, float *S, int N) {
    unsigned int idx = threadIdx.x + blockIdx.x * SECTION_SIZE;
    unsigned int block_id = blockIdx.x;
    if (idx < N && block_id > 0) {
        Y[idx] += S[block_id - 1];
    }
}

// Example usage
void inclusive_scan(float *d_X, float *d_Y, int N) {
    float *d_S;
    int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    cudaMalloc(&d_S, num_blocks * sizeof(float));
    
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(num_blocks);
    local_scan_kernel<<<gridDim, blockDim>>>(d_X, d_Y, d_S, N);
    
    dim3 blockDim2(SECTION_SIZE);
    dim3 gridDim2(1);
    scan_s_kernel<<<gridDim2, blockDim2>>>(d_S, num_blocks);
    
    dim3 blockDim3(SECTION_SIZE);
    dim3 gridDim3(num_blocks);
    add_prefix_kernel<<<gridDim3, blockDim3>>>(d_Y, d_S, N);
    
    cudaFree(d_S);
}