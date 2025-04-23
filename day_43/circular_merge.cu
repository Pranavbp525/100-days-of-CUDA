#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
using namespace std;

// Assuming co_rank and merge_sequential are defined as in the tiled merge kernel code
__device__ int co_rank_circular(int k, int *A, int m, int *B, int n, int A_S_start, int B_S_start, int tile_size) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;
    bool active = true;
    while (active) {
        int i_cir = (A_S_start+i)%tile_size;
        int i_m_1_cir = (A_S_start+i-1)%tile_size;
        int j_cir = (B_S_start+j)%tile_size;
        int j_m_1_cir = (B_S_start+i-1)%tile_size;
        if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

__device__ void merge_sequential_circular(int *A, int m, int *B, int n, int *C, int A_S_start, int B_S_start, int tile_size) {
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < m) && (j < n)) {
        int i_cir =(A_S_start+i)%tile_size;
        int j_cir = (B_S_start+j)%tile_size;
        if (A[i_cir] < B[j_cir]) {
            C[k++] = A[i_cir];i++;
        } else {
            C[k++] = B[j_cir];j++;
        }
    }
    if (i == m) {
        for(;j<n;j++){
            int j_cir = (B_S_start+j)%tile_size;
            C[k++] = B[j_cir];
        }
    } else {
        for(;i<m;i++){
            int i_cir = (A_S_start+i)%tile_size;
            C[k++] = A[i_cir];
        }
    }
}

__global__ void circular_tiled_merge_kernel(int *A, int m, int *B, int n, int *C, int tile_size) {
    extern __shared__ int shareAB[];
    int *A_S = &shareAB[0];
    int *B_S = &shareAB[tile_size];

    int C_curr = blockIdx.x * ceil((float)(m + n) / gridDim.x);
    int C_next = min((blockIdx.x + 1) * ceil((float)(m + n) / gridDim.x), m + n);

    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil((float)C_length / tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;
    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size;
    int B_S_consumed = tile_size;
    
    while (counter < total_iteration) {
        for (int i = 0; i < A_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed && (i + threadIdx.x)<A_S_consumed) {
                A_S[(A_S_start +(tile_size - A_S_consumed)+i+threadIdx.x)%tile_size] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < B_S_consumed; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed && (i + threadIdx.x)<B_S_consumed) {
                B_S[(B_S_start +(tile_size - B_S_consumed)+i+threadIdx.x)%tile_size] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        int a_curr = co_rank_circular(c_curr, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed), A_S_start,B_S_start, tile_size);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next, A_S, min(tile_size, A_length - A_consumed), B_S, min(tile_size, B_length - B_consumed), A_S_start,B_S_start, tile_size);
        int b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next - a_curr, B_S, b_next - b_curr, C + C_curr + C_completed + c_curr, A_S_start+a_curr, B_S_start+b_curr, tile_size);
        counter++;
        A_S_consumed = co_rank_circular(min(tile_size, C_length-C_completed), A_S, min(tile_size, A_length-A_consumed), B_S, min(tile_size, B_lenggth-B_consumed), A_S_start, B_S_start, tile_size)
        B_S_consumed = min(tile_size, C_length-C_completed)-A_S_consumed;
        A_consumed +=A_S_consumed;

        C_completed += min(tile_size, C_length-C_completed);
        
        B_consumed = C_completed - A_consumed;
        A_S_start = (B_S_start + B_S_consumed)%tile_size;
        B_S_start = (A_S_start + B_S_consumed)%tile_size;
        __syncthreads();
    }
}

void parallel_merge_host(int *A_h, int m, int *B_h, int n, int *C_h) {
    int *A_d, *B_d, *C_d;
    size_t size_A = m * sizeof(int);
    size_t size_B = n * sizeof(int);
    size_t size_C = (m + n) * sizeof(int);

    cudaMalloc(&A_d, size_A);
    cudaMalloc(&B_d, size_B);
    cudaMalloc(&C_d, size_C);

    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int tile_size = 1024; // Adjustable parameter based on device capabilities
    int grid_size = (m + n + tile_size - 1) / tile_size;
    if (grid_size == 0) grid_size = 1; // Ensure at least one block

    size_t shared_mem_size = 2 * tile_size * sizeof(int);

    tiled_merge_kernel<<<grid_size, block_size, shared_mem_size>>>(A_d, m, B_d, n, C_d, tile_size);

    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    const int m = 5;
    const int n = 5;
    int A_h[m] = {1, 3, 5, 7, 9};
    int B_h[n] = {2, 4, 6, 8, 10};
    int C_h[m + n];

    parallel_merge_host(A_h, m, B_h, n, C_h);

    cout << "Merged Array: ";
    for (int i = 0; i < m + n; i++) {
        cout << C_h[i] << " ";
    }
    cout << endl;

    return 0;
}