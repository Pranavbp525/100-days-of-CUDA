#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define SETCION_SIZE 512

__global__ void kogge_inclusive_scan_kernel(float * x, float * y, unsigned int N){

    __shared__ float XY[SECTION_SIZE];
    unisgned int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i < N){
        XY[i] = x[i];
    }
    else{
        XY[i] = 0.0f;
    }
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float temp = 0;
        if(threadIdx.x >=stride){
            temp = XY[threadIdx.x] + XY[threadIdx.x-stride]
        }
        __syncthreads()l

    }

}