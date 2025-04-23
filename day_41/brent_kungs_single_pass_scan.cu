#include <cuda_runtime.h>
#include <stdio.h> 

#define SECTION_SIZE 256       
#define THREADS_PER_BLOCK 64 

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)



__device__ float brent_kung_scan(volatile float* array, int tid, int size) {
    for (int stride = 1; stride < size; stride *= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < size) {
            array[idx] += array[idx - stride];
        }
        __syncthreads(); 
    }


    float total_sum = 0.0f;
    if (tid == size -1) {
        total_sum = array[size-1];
        array[size - 1] = 0.0f; 
    }
     __syncthreads(); 


    for (int stride = size / 2; stride > 0; stride /= 2) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < size) {
            float temp = array[idx - stride];
            array[idx - stride] = array[idx];
            array[idx] += temp;
        }
        __syncthreads(); 
    }
    return total_sum; 
}


__global__ void local_scan_kernel_brent_kung(float *X, float *Y, float *S, int N) {
    __shared__ float XY_shared[SECTION_SIZE]; 
    __shared__ float section_sums[THREADS_PER_BLOCK]; 
    __shared__ float block_total_sum_storage; 

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * SECTION_SIZE;
    int subsection_size = SECTION_SIZE / THREADS_PER_BLOCK; 

    float subsection_sum = 0.0f;
    for (int i = 0; i < subsection_size; ++i) {
        int local_idx = tid * subsection_size + i;
        int global_idx = block_offset + local_idx;
        float val = 0.0f;
        if (global_idx < N) {
            val = X[global_idx];
        }
        subsection_sum += val;
        XY_shared[local_idx] = subsection_sum; 
    }
    section_sums[tid] = subsection_sum; 
    __syncthreads();


    float block_total_sum = brent_kung_scan(section_sums, tid, THREADS_PER_BLOCK);

    if (tid == THREADS_PER_BLOCK - 1) {
         block_total_sum_storage = block_total_sum;
    }
    __syncthreads(); 

     if (tid == THREADS_PER_BLOCK - 1) {
         S[blockIdx.x] = block_total_sum_storage;
     }


    for (int i = 0; i < subsection_size; ++i) {
        int local_idx = tid * subsection_size + i;
        int global_idx = block_offset + local_idx;

        if (global_idx < N) {
            
            Y[global_idx] = XY_shared[local_idx] + section_sums[tid];
        }
    }
}


__global__ void scan_s_kernel_brent_kung(float *S, int num_blocks) {
    
    __shared__ float temp[SECTION_SIZE];
    __shared__ float original_s[SECTION_SIZE]; 

    int tid = threadIdx.x;

    if (tid < num_blocks) {
        temp[tid] = S[tid];
    } else {
        temp[tid] = 0.0f; 
    }
     __syncthreads();

    
    brent_kung_scan(temp, tid, SECTION_SIZE); 

    
    if (tid < num_blocks) {
        S[tid] = temp[tid];
    }
}

__global__ void add_prefix_kernel_brent_kung(float *Y, float *S, int N) {
    int block_id = blockIdx.x;
    int local_tid = threadIdx.x;
    int section_start_idx = block_id * SECTION_SIZE;
    int global_idx = section_start_idx + local_tid;

    if (block_id > 0 && global_idx < N) {
        
        Y[global_idx] += S[block_id];
    }
}


void inclusive_scan_brent_kung(float *d_X, float *d_Y, int N) {
    float *d_S; 
    int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    CUDA_CHECK(cudaMalloc(&d_S, num_blocks * sizeof(float)));

    dim3 blockDimLocal(THREADS_PER_BLOCK);
    dim3 gridDimLocal(num_blocks);
    local_scan_kernel_brent_kung<<<gridDimLocal, blockDimLocal>>>(d_X, d_Y, d_S, N);
    CUDA_CHECK(cudaGetLastError()); 

    
    if (num_blocks > 0) {
         if (num_blocks > SECTION_SIZE) {
              fprintf(stderr, "Warning: num_blocks (%d) > SECTION_SIZE (%d). scan_s_kernel needs adaptation for larger inputs.\n", num_blocks, SECTION_SIZE);
              
         }
        dim3 blockDimScanS(SECTION_SIZE); 
        dim3 gridDimScanS(1);       
        scan_s_kernel_brent_kung<<<gridDimScanS, blockDimScanS>>>(d_S, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }


    
    dim3 blockDimAdd(SECTION_SIZE); 
    dim3 gridDimAdd(num_blocks);
    add_prefix_kernel_brent_kung<<<gridDimAdd, blockDimAdd>>>(d_Y, d_S, N);
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cudaFree(d_S));
}


int main() {
    int N = 300; 
    size_t bytes = N * sizeof(float);

    float *h_X = (float *)malloc(bytes);
    float *h_Y = (float *)malloc(bytes);
    float *h_Y_cpu = (float *)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_X[i] = 1.0f;
    }

    float *d_X, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

    inclusive_scan_brent_kung(d_X, d_Y, N);

    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += h_X[i];
        h_Y_cpu[i] = sum;
    }

    int errors = 0;
    float tolerance = 1e-5;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_Y[i] - h_Y_cpu[i]) > tolerance) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("Success! GPU scan matches CPU scan.\n");
        
    } else {
        printf("Failure! %d mismatches found.\n", errors);
    }

    free(h_X);
    free(h_Y);
    free(h_Y_cpu);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    return 0;
}