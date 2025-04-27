#include <iostream>
#include <cuda_runtime.h>
using namespace std;

struct JDSMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int maxNnzPerRow;
    unsigned int* perm;       
    unsigned int* nnzPerRow;  
    unsigned int* colIdx;     
    float* values;            
    unsigned int* jdPtr;      
};

__global__ void spmv_jds_kernel(JDSMatrix jdsMatrix, float *x, float *y) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < jdsMatrix.numRows) {
        float sum = 0.0f;
        unsigned int perm_row = jdsMatrix.perm[row];
        for (unsigned int d = 0; d < jdsMatrix.nnzPerRow[row]; ++d) {
            unsigned int idx = jdsMatrix.jdPtr[d] + row;
            unsigned int col = jdsMatrix.colIdx[idx];
            float value = jdsMatrix.values[idx];
            sum += x[col] * value;
        }
        y[perm_row] = sum;
    }
}

void spmv_jds_host(JDSMatrix jdsMatrix_h, float* x_h, float* y_h) {
    unsigned int *d_perm, *d_nnzPerRow, *d_colIdx, *d_jdPtr;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_perm, jdsMatrix_h.numRows * sizeof(unsigned int));
    cudaMalloc(&d_nnzPerRow, jdsMatrix_h.numRows * sizeof(unsigned int));
    cudaMalloc(&d_colIdx, jdsMatrix_h.numRows * jdsMatrix_h.maxNnzPerRow * sizeof(unsigned int));
    cudaMalloc(&d_values, jdsMatrix_h.numRows * jdsMatrix_h.maxNnzPerRow * sizeof(float));
    cudaMalloc(&d_jdPtr, (jdsMatrix_h.maxNnzPerRow + 1) * sizeof(unsigned int));
    cudaMalloc(&d_x, jdsMatrix_h.numCols * sizeof(float));
    cudaMalloc(&d_y, jdsMatrix_h.numRows * sizeof(float));

    cudaMemcpy(d_perm, jdsMatrix_h.perm, jdsMatrix_h.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzPerRow, jdsMatrix_h.nnzPerRow, jdsMatrix_h.numRows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, jdsMatrix_h.colIdx, jdsMatrix_h.numRows * jdsMatrix_h.maxNnzPerRow * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, jdsMatrix_h.values, jdsMatrix_h.numRows * jdsMatrix_h.maxNnzPerRow * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jdPtr, jdsMatrix_h.jdPtr, (jdsMatrix_h.maxNnzPerRow + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_h, jdsMatrix_h.numCols * sizeof(float), cudaMemcpyHostToDevice);

    JDSMatrix jdsMatrix_d;
    jdsMatrix_d.numRows = jdsMatrix_h.numRows;
    jdsMatrix_d.numCols = jdsMatrix_h.numCols;
    jdsMatrix_d.maxNnzPerRow = jdsMatrix_h.maxNnzPerRow;
    jdsMatrix_d.perm = d_perm;
    jdsMatrix_d.nnzPerRow = d_nnzPerRow;
    jdsMatrix_d.colIdx = d_colIdx;
    jdsMatrix_d.values = d_values;
    jdsMatrix_d.jdPtr = d_jdPtr;

    JDSMatrix *d_jdsMatrix;
    cudaMalloc(&d_jdsMatrix, sizeof(JDSMatrix));
    cudaMemcpy(d_jdsMatrix, &jdsMatrix_d, sizeof(JDSMatrix), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (jdsMatrix_h.numRows + blockSize - 1) / blockSize;

    spmv_jds_kernel<<<numBlocks, blockSize>>>(*d_jdsMatrix, d_x, d_y);

    cudaDeviceSynchronize();

    cudaMemcpy(y_h, d_y, jdsMatrix_h.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_perm);
    cudaFree(d_nnzPerRow);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_jdPtr);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_jdsMatrix);
}

int main() {
    // Example: 2x2 matrix in JDS format
    // Matrix: [1 2]
    //         [0 3]
    unsigned int numRows = 2;
    unsigned int numCols = 2;
    unsigned int maxNnzPerRow = 2;
    unsigned int perm_h[] = {0, 1};  
    unsigned int nnzPerRow_h[] = {2, 1};
    unsigned int colIdx_h[] = {0, 1, 1, 0};  
    float values_h[] = {1.0, 3.0, 2.0, 0.0};  
    unsigned int jdPtr_h[] = {0, 2, 3};  

    float x_h[] = {1.0, 2.0};

    float y_h[2];

    JDSMatrix jdsMatrix_h;
    jdsMatrix_h.numRows = numRows;
    jdsMatrix_h.numCols = numCols;
    jdsMatrix_h.maxNnzPerRow = maxNnzPerRow;
    jdsMatrix_h.perm = perm_h;
    jdsMatrix_h.nnzPerRow = nnzPerRow_h;
    jdsMatrix_h.colIdx = colIdx_h;
    jdsMatrix_h.values = values_h;
    jdsMatrix_h.jdPtr = jdPtr_h;

    spmv_jds_host(jdsMatrix_h, x_h, y_h);

    cout << "Result vector y:" << endl;
    for (int i = 0; i < numRows; i++) {
        cout << y_h[i] << " ";
    }
    cout << endl;

    return 0;
}