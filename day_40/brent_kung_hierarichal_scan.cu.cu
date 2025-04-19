#include <iostream>
#include <cmath> 
#include <vector> 
#include <numeric> 
#include <cmath>
#include <cuda_runtime.h>  

using namespace std;

#define SECTION_SIZE 1024       
#define THREADS_PER_BLOCK 256   

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ \
             << ": " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}


__global__ void local_scan_brent_kung_kernel(float *X, float *Y, float *S, unsigned int N) {
    
    extern __shared__ float shared_mem[]; 
    float* XY = shared_mem; 
    float* temp_sums = &XY[SECTION_SIZE]; 

    unsigned int tid = threadIdx.x;
    unsigned int block_offset = blockIdx.x * SECTION_SIZE;
    unsigned int subsection_size = (SECTION_SIZE / THREADS_PER_BLOCK);
    if (SECTION_SIZE % THREADS_PER_BLOCK != 0) {
         
         return;
    }

    float subsection_sum = 0.0f;
    for (unsigned int i = 0; i < subsection_size; ++i) {
        unsigned int local_idx = tid * subsection_size + i;
         if (local_idx >= SECTION_SIZE) continue;

        unsigned int global_idx = block_offset + local_idx;
        if (global_idx < N) {
            XY[local_idx] = X[global_idx]; 
            subsection_sum += XY[local_idx];
            XY[local_idx] = subsection_sum; 
        } else {
            XY[local_idx] = 0.0f; 
        }
    }

    if (tid < THREADS_PER_BLOCK) {
        temp_sums[tid] = subsection_sum;
    }
    __syncthreads(); 

    
    unsigned int log2_threads = static_cast<unsigned int>(log2f(static_cast<float>(THREADS_PER_BLOCK)));
    for (unsigned int d = 0; d < log2_threads; ++d) {
        unsigned int stride = 1 << d;     
        unsigned int mask = (1 << (d + 1)) - 1; 
        __syncthreads(); 
        if (tid < THREADS_PER_BLOCK && (tid & mask) == mask) { 
             if (tid >= stride) { 
                 temp_sums[tid] += temp_sums[tid - stride];
             }
        }
    }

    if (tid == THREADS_PER_BLOCK - 1) {
        
        if(blockIdx.x < gridDim.x) { 
             S[blockIdx.x] = temp_sums[tid];
        }
        temp_sums[tid] = 0.0f; 
    }
    __syncthreads();


    for (int d = log2_threads - 1; d >= 0; --d) { 
        unsigned int stride = 1 << d;     
        unsigned int mask = (1 << (d + 1)) - 1; 
         __syncthreads(); 
        if (tid < THREADS_PER_BLOCK && (tid & mask) == mask) { 
            if (tid >= stride) { 
                
                float left_val = temp_sums[tid - stride];
                float right_val = temp_sums[tid];
                temp_sums[tid - stride] = right_val; 
                temp_sums[tid] = left_val + right_val; 
            }
        }
    }
    __syncthreads();

    
    float prefix_sum = 0.0f;
    if (tid < THREADS_PER_BLOCK) { 
        prefix_sum = temp_sums[tid]; 
    }

    for (unsigned int i = 0; i < subsection_size; ++i) {
        unsigned int local_idx = tid * subsection_size + i;
        if (local_idx >= SECTION_SIZE) continue;

        unsigned int global_idx = block_offset + local_idx;
        if (global_idx < N) {
            
            Y[global_idx] = XY[local_idx] + prefix_sum;
        }
    }
}


__global__ void scan_s_brent_kung_kernel(float *S, unsigned int num_blocks) {
    extern __shared__ float shared_mem[];
    float* temp_in = shared_mem;
    float* temp_scan = &shared_mem[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int N_scan = num_blocks; 
    unsigned int log2_dim = static_cast<unsigned int>(log2f(static_cast<float>(blockDim.x)));

    if (tid < N_scan) {
        temp_in[tid] = S[tid];
        temp_scan[tid] = S[tid]; 
    } else if (tid < blockDim.x) { 
        temp_in[tid] = 0.0f; 
        temp_scan[tid] = 0.0f;
    }
    __syncthreads();


    for (unsigned int d = 0; d < log2_dim; ++d) {
        unsigned int stride = 1 << d;     
        unsigned int mask = (1 << (d + 1)) - 1; 
        __syncthreads();
        if ((tid & mask) == mask) {
             if (tid >= stride) { 
                 temp_scan[tid] += temp_scan[tid - stride];
             }
        }
    }

    if (tid == blockDim.x - 1) {
        temp_scan[tid] = 0.0f;
    }
    __syncthreads(); 


    for (int d = log2_dim - 1; d >= 0; --d) { 
        unsigned int stride = 1 << d;     
        unsigned int mask = (1 << (d + 1)) - 1; 
        __syncthreads();
        if ((tid & mask) == mask) {
             if (tid >= stride) { 
                 float left_val = temp_scan[tid - stride];
                 float right_val = temp_scan[tid];
                 temp_scan[tid - stride] = right_val;
                 temp_scan[tid] = left_val + right_val;
             }
        }
    }
    __syncthreads(); 
    if (tid < N_scan) {
        S[tid] = temp_scan[tid] + temp_in[tid];
    }
}


__global__ void add_prefix_kernel(float *Y, float *S, unsigned int N) {
    unsigned int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int block_id = global_idx / SECTION_SIZE;
    if (global_idx < N && block_id > 0 && (block_id - 1) < gridDim.x ) {
        Y[global_idx] += S[block_id - 1];
    }
}

void hierarchical_scan_brent_kung_host(float *X_h, float *Y_h, unsigned int N) {
     if (N == 0) {
        cout << "Input size N cannot be 0." << endl;
        return;
    }
    if (SECTION_SIZE == 0 || THREADS_PER_BLOCK == 0) {
        cerr << "Error: SECTION_SIZE and THREADS_PER_BLOCK must be > 0" << endl;
        return;
    }
    if (SECTION_SIZE % THREADS_PER_BLOCK != 0) {
        cerr << "Error: SECTION_SIZE must be divisible by THREADS_PER_BLOCK" << endl;
        return;
    }
    if ((THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) != 0) {
        cerr << "Error: THREADS_PER_BLOCK must be a power of 2 for this Brent-Kung implementation." << endl;
        return;
    }


    unsigned int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;

    float *X_d, *Y_d, *S_d;

    CUDA_CHECK(cudaMalloc(&X_d, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Y_d, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S_d, num_blocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim1(num_blocks);
    dim3 blockDim1(THREADS_PER_BLOCK);
    size_t local_scan_shared_mem_size = (SECTION_SIZE + THREADS_PER_BLOCK) * sizeof(float);
    cout << "Launching local_scan_brent_kung_kernel with " << gridDim1.x << " blocks, " << blockDim1.x << " threads, " << local_scan_shared_mem_size << " bytes shared mem..." << endl;
    local_scan_brent_kung_kernel<<<gridDim1, blockDim1, local_scan_shared_mem_size>>>(X_d, Y_d, S_d, N);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize()); 

    if (num_blocks > 1) { 
        dim3 gridDim2(1); 
        unsigned int scan_s_threads = 1;
        while(scan_s_threads < num_blocks) scan_s_threads *= 2;
        scan_s_threads = min(scan_s_threads, (unsigned int)1024); 
         if (scan_s_threads < num_blocks) {
              cerr << "Error: Cannot scan block sums, num_blocks (" << num_blocks
                   << ") exceeds adjusted thread count (" << scan_s_threads << ")" << endl;
              cudaFree(X_d); cudaFree(Y_d); cudaFree(S_d);
              return;
         }

        dim3 blockDim2(scan_s_threads);
        size_t scan_s_shared_mem_size = 2 * scan_s_threads * sizeof(float);

        cout << "Launching scan_s_brent_kung_kernel with " << gridDim2.x << " block, " << blockDim2.x << " threads (processing " << num_blocks << " sums), " << scan_s_shared_mem_size << " bytes shared mem..." << endl;
        scan_s_brent_kung_kernel<<<gridDim2, blockDim2, scan_s_shared_mem_size>>>(S_d, num_blocks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else if (num_blocks == 1) {
        
    }



    unsigned int threads_add = THREADS_PER_BLOCK; 
    unsigned int blocks_add = (N + threads_add - 1) / threads_add;
    dim3 gridDim3(blocks_add);
    dim3 blockDim3(threads_add);

    cout << "Launching add_prefix_kernel with " << gridDim3.x << " blocks and " << blockDim3.x << " threads..." << endl;
    add_prefix_kernel<<<gridDim3, blockDim3>>>(Y_d, S_d, N); 
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    CUDA_CHECK(cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(X_d));
    CUDA_CHECK(cudaFree(Y_d));
    CUDA_CHECK(cudaFree(S_d));
}

int main() {
    const unsigned int N = 4096;
    vector<float> X_h(N);
    vector<float> Y_h(N);
    vector<float> Y_expected(N);

    for (unsigned int i = 0; i < N; i++) {
        X_h[i] = 1.0f;
    }

    cout << "Running Hierarchical Scan with Brent-Kung..." << endl;
    hierarchical_scan_brent_kung_host(X_h.data(), Y_h.data(), N);
    cout << "Scan finished." << endl;

    cout << "Verifying results..." << endl;
    partial_sum(X_h.begin(), X_h.end(), Y_expected.begin());

    bool success = true;
    unsigned int errors = 0;
    double max_error = 0.0;
    const double tolerance = 1e-5; 

    for (unsigned int i = 0; i < N; i++) {
        double diff = abs(static_cast<double>(Y_h[i]) - static_cast<double>(Y_expected[i]));
        if (diff > tolerance) {
             if (errors < 10) { 
                cout.precision(8); 
                cout << "Verification Failed at index " << i
                     << ": Expected=" << fixed << Y_expected[i]
                     << ", Got=" << fixed << Y_h[i]
                     << ", Diff=" << fixed << diff << endl;
             }
             success = false;
             errors++;
             max_error = max(max_error, diff);
        }
    }

    if (success) {
         cout << "Verification Successful!" << endl;
    } else {
         cout << "Verification Failed! Total errors: " << errors << ", Max error: " << max_error << endl;
    }

    cout << "GPU Scan Result (elements 1020 to 1030):" << endl;
    cout.precision(1); 
    for (unsigned int i = 1020; i < 1031 && i < N; i++) {
        cout << "Y[" << i << "]=" << fixed << Y_h[i] << " ";
    }
    cout << endl;

    return success ? 0 : 1; 
}