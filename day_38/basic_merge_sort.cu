#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__device__ int co_rank(int k, int *A, int m, int *B, int n) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
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

__device__ void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0;
    int j = 0;
    int k = 0;
    while ((i < m) && (j < n)) {
        if (A[i] < B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {
        while (j < n) {
            C[k++] = B[j++];
        }
    } else {
        while (i < m) {
            C[k++] = A[i++];
        }
    }
}

__global__ void basic_merge_kernel(int *A, int m, int *B, int n, int *C) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int elementsPerThread = ceil((m + n) / (float)(blockDim.x * gridDim.x));
    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, m + n);
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
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

    int block_size = 256;  
    int grid_size = (m + n + block_size - 1) / block_size;  


    basic_merge_kernel<<<grid_size, block_size>>>(A_d, m, B_d, n, C_d);

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