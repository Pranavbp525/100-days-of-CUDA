#include <iostream>
#include <cuda_runtime.h>
using namespace std;


struct CSRMatrix {
    unsigned int* rowPtrs;     
    unsigned int* colIdx;      
    float* values;             
    unsigned int numRows;      
    unsigned int numCols;      
    unsigned int numNonZeros;  
};

__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float *x, float *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < csrMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {
            unsigned int col = csrMatrix.colIdx[i];
            float value = csrMatrix.values[i];  
            sum += x[col] * value;
        }
        y[row] += sum;  
    }
}

void spmv_csr_host(CSRMatrix csrMatrix_h, float* x_h, float* y_h) {
    unsigned int *d_rowPtrs, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtrs, (csrMatrix_h.numRows + 1) * sizeof(unsigned int));
    cudaMalloc(&d_colIdx, csrMatrix_h.numNonZeros * sizeof(unsigned int));
    cudaMalloc(&d_values, csrMatrix_h.numNonZeros * sizeof(float));
    cudaMalloc(&d_x, csrMatrix_h.numCols * sizeof(float));
    cudaMalloc(&d_y, csrMatrix_h.numRows * sizeof(float));

    cudaMemcpy(d_rowPtrs, csrMatrix_h.rowPtrs, (csrMatrix_h.numRows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, csrMatrix_h.colIdx, csrMatrix_h.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, csrMatrix_h.values, csrMatrix_h.numNonZeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_h, csrMatrix_h.numCols * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_y, y_h, csrMatrix_h.numRows * sizeof(float), cudaMemcpyHostToDevice);

    CSRMatrix csrMatrix_d;
    csrMatrix_d.rowPtrs = d_rowPtrs;
    csrMatrix_d.colIdx = d_colIdx;
    csrMatrix_d.values = d_values;
    csrMatrix_d.numRows = csrMatrix_h.numRows;
    csrMatrix_d.numCols = csrMatrix_h.numCols;
    csrMatrix_d.numNonZeros = csrMatrix_h.numNonZeros;

    CSRMatrix *d_csrMatrix;
    cudaMalloc(&d_csrMatrix, sizeof(CSRMatrix));
    cudaMemcpy(d_csrMatrix, &csrMatrix_d, sizeof(CSRMatrix), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (csrMatrix_h.numRows + blockSize - 1) / blockSize;

    spmv_csr_kernel<<<numBlocks, blockSize>>>(*d_csrMatrix, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(y_h, d_y, csrMatrix_h.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtrs);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_csrMatrix);
}

int main() {
    
    // Matrix: [1 2]
    //         [0 3]
    unsigned int rowPtrs_h[] = {0, 2, 3};  
    unsigned int colIdx_h[] = {0, 1, 1};   
    float values_h[] = {1.0, 2.0, 3.0};    
    unsigned int numRows = 2;
    unsigned int numCols = 2;
    unsigned int numNonZeros = 3;

    float x_h[] = {1.0, 2.0};  

    float y_h[2] = {1.0, 1.0};  
    CSRMatrix csrMatrix_h;
    csrMatrix_h.rowPtrs = rowPtrs_h;
    csrMatrix_h.colIdx = colIdx_h;
    csrMatrix_h.values = values_h;
    csrMatrix_h.numRows = numRows;
    csrMatrix_h.numCols = numCols;
    csrMatrix_h.numNonZeros = numNonZeros;

    spmv_csr_host(csrMatrix_h, x_h, y_h);

    cout << "Result vector y:" << endl;
    for (int i = 0; i < numRows; i++) {
        cout << y_h[i] << " ";
    }
    cout << endl;

    return 0;
}
