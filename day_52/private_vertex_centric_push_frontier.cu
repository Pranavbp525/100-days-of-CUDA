#include <iostream>
#include <climits> 
#include <cuda_runtime.h>
using namespace std;

struct CSRGraph {
    unsigned int numVertices;
    unsigned int* srcPtrs;  
    unsigned int* dst;      
};

__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int *prevFrontier, unsigned int *currFrontier, unsigned int* numprevFrontier, unsigned int* numcurrFrontier, unsigned int currLevel) {
    const unsigned int LOCAL_FRONTIER_CAPACITY = 1024;  
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < *numprevFrontier) {  
        unsigned int vertex = prevFrontier[i];
         
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int currFrontierIdx = atomicAdd(numcurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }

    __syncthreads();

    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numcurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    for (unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s; currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

void bfs_host(CSRGraph csrGraph_h, unsigned int startVertex, unsigned int* level_h) {
    unsigned int *d_srcPtrs, *d_dst, *d_level, *d_prevFrontier, *d_currFrontier, *d_numprevFrontier, *d_numcurrFrontier;
    unsigned int numVertices = csrGraph_h.numVertices;

    cudaMalloc(&d_srcPtrs, (numVertices + 1) * sizeof(unsigned int));
    cudaMalloc(&d_dst, csrGraph_h.srcPtrs[numVertices] * sizeof(unsigned int));  
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));

    cudaMalloc(&d_prevFrontier, numVertices * sizeof(unsigned int));
    cudaMalloc(&d_currFrontier, numVertices * sizeof(unsigned int));
    cudaMalloc(&d_numprevFrontier, sizeof(unsigned int));
    cudaMalloc(&d_numcurrFrontier, sizeof(unsigned int));

    cudaMemcpy(d_srcPtrs, csrGraph_h.srcPtrs, (numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, csrGraph_h.dst, csrGraph_h.srcPtrs[numVertices] * sizeof(unsigned int), cudaMemcpyHostToDevice);

    for (unsigned int i = 0; i < numVertices; i++) {
        level_h[i] = (i == startVertex) ? 0 : UINT_MAX;
    }
    cudaMemcpy(d_level, level_h, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);

    CSRGraph csrGraph_d;
    csrGraph_d.numVertices = numVertices;
    csrGraph_d.srcPtrs = d_srcPtrs;
    csrGraph_d.dst = d_dst;

    CSRGraph *d_csrGraph;
    cudaMalloc(&d_csrGraph, sizeof(CSRGraph));
    cudaMemcpy(d_csrGraph, &csrGraph_d, sizeof(CSRGraph), cudaMemcpyHostToDevice);

    unsigned int h_numprevFrontier = 1;
    unsigned int h_numcurrFrontier = 0;
    unsigned int h_prevFrontier[1] = {startVertex};
    cudaMemcpy(d_prevFrontier, h_prevFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numprevFrontier, &h_numprevFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numcurrFrontier, &h_numcurrFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int currLevel = 1;
    while (h_numprevFrontier > 0) {
        h_numcurrFrontier = 0;
        cudaMemcpy(d_numcurrFrontier, &h_numcurrFrontier, sizeof(unsigned int), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (h_numprevFrontier + blockSize - 1) / blockSize;

        bfs_kernel<<<numBlocks, blockSize>>>(*d_csrGraph, d_level, d_prevFrontier, d_currFrontier, d_numprevFrontier, d_numcurrFrontier, currLevel);
        cudaDeviceSynchronize();  

        unsigned int* temp = d_prevFrontier;
        d_prevFrontier = d_currFrontier;
        d_currFrontier = temp;

        cudaMemcpy(&h_numprevFrontier, d_numcurrFrontier, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        currLevel++;
    }

    cudaMemcpy(level_h, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_level);
    cudaFree(d_prevFrontier);
    cudaFree(d_currFrontier);
    cudaFree(d_numprevFrontier);
    cudaFree(d_numcurrFrontier);
    cudaFree(d_csrGraph);
}

int main() {
    unsigned int numVertices = 4;
    unsigned int srcPtrs_h[] = {0, 1, 2, 3, 4};  
    unsigned int dst_h[] = {1, 2, 3, 0};         

    CSRGraph csrGraph_h;
    csrGraph_h.numVertices = numVertices;
    csrGraph_h.srcPtrs = srcPtrs_h;
    csrGraph_h.dst = dst_h;

    unsigned int startVertex = 0;
    unsigned int level_h[4];

    bfs_host(csrGraph_h, startVertex, level_h);

    for (unsigned int i = 0; i < numVertices; i++) {
        cout << "Vertex " << i << ": Level " << level_h[i] << endl;
    }

    return 0;
}