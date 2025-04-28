#include <iostream>
#include <climits> 
#include <cuda_runtime.h>
using namespace std;

struct CSCGraph {
    unsigned int numVertices;
    unsigned int* dstPtrs;  
    unsigned int* src;      
};

__global__ void bfs_kernel(CSCGraph cscGraph, unsigned int* level, unsigned int *newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertex < cscGraph.numVertices) {
        if (level[vertex] == UINT_MAX) {  
            for (unsigned int edge = cscGraph.dstPtrs[vertex]; edge < cscGraph.dstPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = cscGraph.src[edge];
                if (level[neighbor] == currLevel - 1) {
                    level[vertex] = currLevel;
                    atomicExch(newVertexVisited, 1);
                    break;
                }
            }
        }
    }
}

void bfs_host(CSCGraph cscGraph_h, unsigned int startVertex, unsigned int* level_h) {
    unsigned int *d_dstPtrs, *d_src, *d_level, *d_newVertexVisited;
    unsigned int numVertices = cscGraph_h.numVertices;

    cudaMalloc(&d_dstPtrs, (numVertices + 1) * sizeof(unsigned int));
    cudaMalloc(&d_src, cscGraph_h.dstPtrs[numVertices] * sizeof(unsigned int));
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

    cudaMemcpy(d_dstPtrs, cscGraph_h.dstPtrs, (numVertices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, cscGraph_h.src, cscGraph_h.dstPtrs[numVertices] * sizeof(unsigned int), cudaMemcpyHostToDevice);

    for (unsigned int i = 0; i < numVertices; i++) {
        level_h[i] = (i == startVertex) ? 0 : UINT_MAX;
    }
    cudaMemcpy(d_level, level_h, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);

    CSCGraph cscGraph_d;
    cscGraph_d.numVertices = numVertices;
    cscGraph_d.dstPtrs = d_dstPtrs;
    cscGraph_d.src = d_src;

    CSCGraph *d_cscGraph;
    cudaMalloc(&d_cscGraph, sizeof(CSCGraph));
    cudaMemcpy(d_cscGraph, &cscGraph_d, sizeof(CSCGraph), cudaMemcpyHostToDevice);

    unsigned int currLevel = 1;
    unsigned int newVertexVisited_h = 1;
    while (newVertexVisited_h) {
        cudaMemset(d_newVertexVisited, 0, sizeof(unsigned int));

        int blockSize = 256;
        int numBlocks = (numVertices + blockSize - 1) / blockSize;

        bfs_kernel<<<numBlocks, blockSize>>>(*d_cscGraph, d_level, d_newVertexVisited, currLevel);
        cudaDeviceSynchronize();

        cudaMemcpy(&newVertexVisited_h, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        currLevel++;
    }

    cudaMemcpy(level_h, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_dstPtrs);
    cudaFree(d_src);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);
    cudaFree(d_cscGraph);
}

int main() {
    unsigned int numVertices = 4;
    unsigned int dstPtrs_h[] = {0, 0, 1, 2, 3};
    unsigned int src_h[] = {0, 1, 2};

    CSCGraph cscGraph_h;
    cscGraph_h.numVertices = numVertices;
    cscGraph_h.dstPtrs = dstPtrs_h;
    cscGraph_h.src = src_h;

    unsigned int startVertex = 0;
    unsigned int level_h[4];

    bfs_host(cscGraph_h, startVertex, level_h);

    for (unsigned int i = 0; i < numVertices; i++) {
        cout << "Vertex " << i << ": Level " << level_h[i] << endl;
    }

    return 0;
}