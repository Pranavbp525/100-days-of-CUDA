#include <iostream>
#include <climits> 
#include <cuda_runtime.h>
using namespace std;

struct COOGraph {
    unsigned int numEdges;
    unsigned int numVertices;
    unsigned int* src;  
    unsigned int* dst;  
};

__global__ void bfs_kernel(COOGraph cooGraph, unsigned int *level, unsigned int *newVertexVisited, unsigned int currLevel) {
    unsigned int edge = threadIdx.x + blockIdx.x * blockDim.x;
    if (edge < cooGraph.numEdges) {  
        unsigned int vertex = cooGraph.src[edge];
        if (level[vertex] == currLevel - 1) {  
            unsigned int neighbor = cooGraph.dst[edge];
            if (level[neighbor] == UINT_MAX) {
                level[neighbor] = currLevel;
                *newVertexVisited = 1;  
            }
        }
    }
}

void bfs_host(COOGraph cooGraph_h, unsigned int startVertex, unsigned int* level_h) {
    unsigned int *d_src, *d_dst, *d_level, *d_newVertexVisited;
    unsigned int numEdges = cooGraph_h.numEdges;
    unsigned int numVertices = cooGraph_h.numVertices;

    cudaMalloc(&d_src, numEdges * sizeof(unsigned int));
    cudaMalloc(&d_dst, numEdges * sizeof(unsigned int));
    cudaMalloc(&d_level, numVertices * sizeof(unsigned int));
    cudaMalloc(&d_newVertexVisited, sizeof(unsigned int));

    cudaMemcpy(d_src, cooGraph_h.src, numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, cooGraph_h.dst, numEdges * sizeof(unsigned int), cudaMemcpyHostToDevice);

    for (unsigned int i = 0; i < numVertices; i++) {
        level_h[i] = (i == startVertex) ? 0 : UINT_MAX;
    }
    cudaMemcpy(d_level, level_h, numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice);

    COOGraph cooGraph_d;
    cooGraph_d.numEdges = numEdges;
    cooGraph_d.numVertices = numVertices;
    cooGraph_d.src = d_src;
    cooGraph_d.dst = d_dst;

    COOGraph *d_cooGraph;
    cudaMalloc(&d_cooGraph, sizeof(COOGraph));
    cudaMemcpy(d_cooGraph, &cooGraph_d, sizeof(COOGraph), cudaMemcpyHostToDevice);

    unsigned int currLevel = 1;
    unsigned int newVertexVisited_h = 1;
    while (newVertexVisited_h) {
        cudaMemset(d_newVertexVisited, 0, sizeof(unsigned int));

        int blockSize = 256;
        int numBlocks = (numEdges + blockSize - 1) / blockSize;

        bfs_kernel<<<numBlocks, blockSize>>>(*d_cooGraph, d_level, d_newVertexVisited, currLevel);
        cudaDeviceSynchronize();  

        cudaMemcpy(&newVertexVisited_h, d_newVertexVisited, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        currLevel++;
    }

    cudaMemcpy(level_h, d_level, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_level);
    cudaFree(d_newVertexVisited);
    cudaFree(d_cooGraph);
}

int main() {
    unsigned int numVertices = 4;
    unsigned int numEdges = 3;
    unsigned int src_h[] = {0, 1, 2};  
    unsigned int dst_h[] = {1, 2, 3};  

    COOGraph cooGraph_h;
    cooGraph_h.numEdges = numEdges;
    cooGraph_h.numVertices = numVertices;
    cooGraph_h.src = src_h;
    cooGraph_h.dst = dst_h;

    unsigned int startVertex = 0;
    unsigned int level_h[4];

    bfs_host(cooGraph_h, startVertex, level_h);

    for (unsigned int i = 0; i < numVertices; i++) {
        cout << "Vertex " << i << ": Level " << level_h[i] << endl;
    }

    return 0;
}