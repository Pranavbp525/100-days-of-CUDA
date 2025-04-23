#include <iostream>
#include <cuda_runtime.h>
using namespace std;

struct COOMatrix {
    unsigned int* rowIdx;      
    unsigned int* colIdx;      
    float* values;             
    unsigned int numNonZeros;  
    unsigned int numRows;      
    unsigned int numCols;      
};

__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float *x, float *y) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < cooMatrix.numNonZeros) {
        unsigned int row = cooMatrix.rowIdx[i];
        unsigned int col = cooMatrix.colIdx[i];  
        float value = cooMatrix.values[i];       
        atomicAdd(&y[row], x[col] * value);
    }
}

void spmv_coo_host(COOMatrix cooMatrix_h, float* x_h, float* y_h) {
    unsigned int *d_rowIdx, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowIdx, cooMatrix_h.numNonZeros * sizeof(unsigned int));
    cudaMalloc(&d_colIdx, cooMatrix_h.numNonZeros * sizeof(unsigned int));
    cudaMalloc(&d_values, cooMatrix_h.numNonZeros * sizeof(float));
    cudaMalloc(&d_x, cooMatrix_h.numCols * sizeof(float));
    cudaMalloc(&d_y, cooMatrix_h.numRows * sizeof(float));

    cudaMemcpy(d_rowIdx, cooMatrix_h.rowIdx, cooMatrix_h.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, cooMatrix_h.colIdx, cooMatrix_h.numNonZeros * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, cooMatrix_h.values, cooMatrix_h.numNonZeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_h, cooMatrix_h.numCols * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_y, 0, cooMatrix_h.numRows * sizeof(float));

    COOMatrix cooMatrix_d;
    cooMatrix_d.rowIdx = d_rowIdx;
    cooMatrix_d.colIdx = d_colIdx;
    cooMatrix_d.values = d_values;
    cooMatrix_d.numNonZeros = cooMatrix_h.numNonZeros;
    cooMatrix_d.numRows = cooMatrix_h.numRows;
    cooMatrix_d.numCols = cooMatrix_h.numCols;

    COOMatrix *d_cooMatrix;
    cudaMalloc(&d_cooMatrix, sizeof(COOMatrix));
    cudaMemcpy(d_cooMatrix, &cooMatrix_d, sizeof(COOMatrix), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (cooMatrix_h.numNonZeros + blockSize - 1) / blockSize;

    spmv_coo_kernel<<<numBlocks, blockSize>>>(*d_cooMatrix, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(y_h, d_y, cooMatrix_h.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rowIdx);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_cooMatrix);
}

int main() {
    // Example: 2x2 matrix in COO format
    // Matrix: [1 2]
    //         [0 3]
    unsigned int rowIdx_h[] = {0, 0, 1};  
    unsigned int colIdx_h[] = {0, 1, 1};  
    float values_h[] = {1.0, 2.0, 3.0};   
    unsigned int numNonZeros = 3;
    unsigned int numRows = 2;
    unsigned int numCols = 2;

    float x_h[] = {1.0, 2.0};  

    float y_h[2] = {0.0, 0.0};  

    COOMatrix cooMatrix_h;
    cooMatrix_h.rowIdx = rowIdx_h;
    cooMatrix_h.colIdx = colIdx_h;
    cooMatrix_h.values = values_h;
    cooMatrix_h.numNonZeros = numNonZeros;
    cooMatrix_h.numRows = numRows;
    cooMatrix_h.numCols = numCols;

    spmv_coo_host(cooMatrix_h, x_h, y_h);

    cout << "Result vector y:" << endl;
    for (int i = 0; i < numRows; i++) {
        cout << y_h[i] << " ";
    }
    cout << endl;

    return 0;
}