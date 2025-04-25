#include <iostream>
#include <cuda_runtime.h>
using namespace std;

struct EllMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int maxNnzPerRow;
    unsigned int* nnzPerRow;  
    unsigned int* colIdx;     
    float* values;           
};

__global__ void spmv_ell_kernel(EllMatrix ellMatrix, float *x, float *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.nnzPerRow[row]; ++t) {
            unsigned int i = t * ellMatrix.numRows + row;
            unsigned int col = ellMatrix.colIdx[i];
            float value = ellMatrix.values[i];  
            sum += x[col] * value;
        }
        y[row] = sum;
    }
}

void spmv_ell_host(EllMatrix ellMatrix_h, float* x_h, float* y_h) {
    unsigned int *d_nnzPerRow, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_nnzPerRow, ellMatrix_h.numRows * sizeof(unsigned int));
    cudaMalloc(&d_colIdx, ellMatrix_h.maxNnzPerRow * ellMatrix_h.numRows * sizeof(unsigned int));
    cudaMalloc(&d_values, ellMatrix_h.maxNnzPerRow * ellMatrix_h.numRows * sizeof(float));
    cudaMalloc(&d_x, ellMatrix_h.numCols * sizeof(float));
    cudaMalloc(&d_y, ellMatrix_h.numRows * sizeof(float));

    cudaMemcpy(d_nnzPerRow, ellMatrix_h.nnzPerRow, ellMatrix_h.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, ellMatrix_h.colIdx, ellMatrix_h.maxNnzPerRow * ellMatrix_h.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, ellMatrix_h.values, ellMatrix_h.maxNnzPerRow * ellMatrix_h.numRows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_h, ellMatrix_h.numCols * sizeof(float), cudaMemcpyHostToDevice);

    EllMatrix ellMatrix_d;
    ellMatrix_d.numRows = ellMatrix_h.numRows;
    ellMatrix_d.numCols = ellMatrix_h.numCols;
    ellMatrix_d.maxNnzPerRow = ellMatrix_h.maxNnzPerRow;
    ellMatrix_d.nnzPerRow = d_nnzPerRow;
    ellMatrix_d.colIdx = d_colIdx;
    ellMatrix_d.values = d_values;

    EllMatrix *d_ellMatrix;
    cudaMalloc(&d_ellMatrix, sizeof(EllMatrix));
    cudaMemcpy(d_ellMatrix, &ellMatrix_d, sizeof(EllMatrix), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (ellMatrix_h.numRows + blockSize - 1) / blockSize;

    spmv_ell_kernel<<<numBlocks, blockSize>>>(*d_ellMatrix, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(y_h, d_y, ellMatrix_h.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_nnzPerRow);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_ellMatrix);
}

int main() {
    // Example: 2x2 matrix in ELL format
    // Matrix: [1 2]
    //         [0 3]
    unsigned int numRows = 2;
    unsigned int numCols = 2;
    unsigned int maxNnzPerRow = 2;
    unsigned int nnzPerRow_h[] = {2, 1};
    unsigned int colIdx_h[] = {0, 1, 1, 0};  // Padded with 0
    float values_h[] = {1.0, 3.0, 2.0, 0.0};  // Padded with 0.0

    float x_h[] = {1.0, 2.0};

    float y_h[2];

    EllMatrix ellMatrix_h;
    ellMatrix_h.numRows = numRows;
    ellMatrix_h.numCols = numCols;
    ellMatrix_h.maxNnzPerRow = maxNnzPerRow;
    ellMatrix_h.nnzPerRow = nnzPerRow_h;
    ellMatrix_h.colIdx = colIdx_h;
    ellMatrix_h.values = values_h;

    spmv_ell_host(ellMatrix_h, x_h, y_h);

    cout << "Result vector y:" << endl;
    for (int i = 0; i < numRows; i++) {
        cout << y_h[i] << " ";
    }
    cout << endl;

    return 0;
}