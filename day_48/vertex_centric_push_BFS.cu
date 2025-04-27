#include <iostream>
#include <cuda_runtime.h>
#include <climits> 
using namespace std;

struct CSRGraph {
    unsigned int numVertices;
    unsigned int* srcPtrs;  
    unsigned int* dst;      
};

__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int *newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = threadIdx.x + blockIdx.x * blockDim.x;
    if (vertex < csrGraph.numVertices) {
        if (level[vertex] == currLevel - 1) {  
            for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (level[neighbor] == UINT_MAX) {
                    level[neighbor] = currLevel;
                    *newVertexVisited = 1;
                }
            }
        }
    }
}


void bfs_host(CSRGraph csrGraph_h, unsigned int startVertex, unsigned int* level_h) {
    unsigned int *d_srcPtrs, *d_dst, *d_level, *d_newVertexVisited;
    unsigned int numVertices = csrGraph_h.numVertices;

    cudaMalloc(&d_srcPtrs, (numVertices + 1) * sizeof(unsigned int));
    cudaMalloc(&d_dst, csrGraph_h.srcPtrs[numVertices] * sizeof(unsigned int));  // Total edges
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

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

    unsigned int currLevel = 1;
    unsigned int newVertexVisited_h = 1;
    while (newVertexVisited_h) {
        cudaMemset(d_newVertexVisited, 0, sizeof(unsigned int));

        int blockSize = 256;
        int numBlocks = (numVertices + blockSize - 1) / blockSize;

        bfs_kernel<<<numBlocks, blockSize>>>(*d_csrGraph, d_level, d_newVertexVisited, currLevel);
        cudaDeviceSynchronize();  


        cudaMemcpy(&newVertexVisited_h, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        currLevel++;
    }

    cudaMemcpy(level_h, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);
    cudaFree(d_csrGraph);
}

int main() {
    // Example: A simple graph with 4 vertices
    // 0 -> 1 -> 2 -> 3
    unsigned int numVertices = 4;
    unsigned int srcPtrs_h[] = {0, 1, 2, 3, 4};  // Edge offsets
    unsigned int dst_h[] = {1, 2, 3, 0};         // Destinations (edges)

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