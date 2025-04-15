#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define SECTION_SIZE 1024  // Number of elements per block
#define THREADS_PER_BLOCK 256  // Threads per block
#define NUM_BLOCKS 4  // Example for N=4096 with SECTION_SIZE=1024

// CUDA kernel for single-pass scan with dynamic block index assignment and adjacent synchronization
__global__ void single_pass_scan_kernel(float *X, float *Y, int *flags, int *blockCounter, float *scan_value, unsigned int N) {
    __shared__ float XY[SECTION_SIZE];  // Shared memory for local scan
    __shared__ int bid_s;  // Shared memory for dynamic block index

    unsigned int tid = threadIdx.x;
    unsigned int block_offset;

    // Step 1: Dynamic block index assignment
    if (tid == 0) {
        bid_s = atomicAdd(blockCounter, 1);  // Atomically assign block index
    }
    __syncthreads();
    int bid = bid_s;

    // Calculate the starting offset for this block
    block_offset = bid * SECTION_SIZE;

    // Step 2: Load data into shared memory and perform local scan (Phase 1)
    if (block_offset + tid < N) {
        XY[tid] = X[block_offset + tid];
    } else {
        XY[tid] = 0.0f;  // Pad with zeros if beyond array bounds
    }
    __syncthreads();

    // Perform a simple Kogge-Stone scan within the block
    for (unsigned int stride = 1; stride <= THREADS_PER_BLOCK; stride *= 2) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = XY[tid] + XY[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            XY[tid] = temp;
        }
        __syncthreads();
    }

    // Compute the local sum (last element of the local scan)
    float local_sum = XY[THREADS_PER_BLOCK - 1];

    // Step 3: Wait for predecessor and get cumulative sum (Phase 2)
    float previous_sum = 0.0f;
    if (bid > 0) {
        // Wait for predecessor to set the flag
        while (atomicCAS(&flags[bid], 0, 0) == 0) {}
        previous_sum = scan_value[bid];  // Load cumulative sum from predecessor
    }

    // Step 4: Update local results with cumulative sum (Phase 3)
    if (block_offset + tid < N) {
        Y[block_offset + tid] = XY[tid] + previous_sum;
    }

    // Step 5: Pass cumulative sum to successor
    if (bid < NUM_BLOCKS - 1) {
        float block_sum = local_sum + previous_sum;  // Total sum for this block
        scan_value[bid + 1] = block_sum;  // Store for next block
        __threadfence();  // Ensure memory is written before flag is set
        atomicAdd(&flags[bid + 1], 1);  // Signal the next block
    }
}

// Host function to manage memory and launch the kernel
void single_pass_scan_host(float *X_h, float *Y_h, unsigned int N) {
    unsigned int num_blocks = (N + SECTION_SIZE - 1) / SECTION_SIZE;
    float *X_d, *Y_d, *scan_value_d;
    int *flags_d, *blockCounter_d;

    // Allocate device memory
    cudaMalloc(&X_d, N * sizeof(float));
    cudaMalloc(&Y_d, N * sizeof(float));
    cudaMalloc(&scan_value_d, num_blocks * sizeof(float));
    cudaMalloc(&flags_d, num_blocks * sizeof(int));
    cudaMalloc(&blockCounter_d, sizeof(int));

    // Initialize device memory
    cudaMemset(flags_d, 0, num_blocks * sizeof(int));  // Clear flags
    cudaMemset(blockCounter_d, 0, sizeof(int));  // Reset block counter
    cudaMemcpy(X_d, X_h, N * sizeof(float), cudaMemcpyHostToDevice);  // Copy input to device

    // Launch kernel
    dim3 gridDim(num_blocks);
    dim3 blockDim(THREADS_PER_BLOCK);
    single_pass_scan_kernel<<<gridDim, blockDim>>>(X_d, Y_d, flags_d, blockCounter_d, scan_value_d, N);

    // Copy result back to host
    cudaMemcpy(Y_h, Y_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(scan_value_d);
    cudaFree(flags_d);
    cudaFree(blockCounter_d);
}

// Main function to test the scan
int main() {
    const unsigned int N = 4096;  // Total number of elements
    float *X_h = new float[N];    // Host input array
    float *Y_h = new float[N];    // Host output array

    // Initialize input array (e.g., all 1.0f for a simple test)
    for (unsigned int i = 0; i < N; i++) {
        X_h[i] = 1.0f;  // Expected output: 1, 2, 3, ..., N
    }

    // Perform the scan
    single_pass_scan_host(X_h, Y_h, N);

    // Print the first 10 elements to verify
    cout << "Inclusive Scan Result (first 10 elements):" << endl;
    for (unsigned int i = 0; i < 10 && i < N; i++) {
        cout << Y_h[i] << " ";
    }
    cout << endl;

    // Clean up host memory
    delete[] X_h;
    delete[] Y_h;

    return 0;
}