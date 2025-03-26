#incude <iostream>
#include <cuda_runtime.h>
using namespace std;

#define IN_TILE_DIM 5
#define OUT_TILE_DIM 4

__global__ void tiled_stencil(float * inp, float * out, int side){

    int i = threadIdx.z + blockIdx.z*OUT_TILE_DIM - 1;
    int j = threadIdx.y + blockIdx.y*OUT_TILE_DIM - 1;
    int k = threadIdx.x + blockIdx.x*OUT_TILE_DIM - 1;

    __shared__ float inp_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if(i>=0 && i<side && j>=0 && j<side && k>=0 && k<side){
        inp_s[threadIdx.z][threadIdx.y][threadIdx.x] = inp[i*side*side + j*side + k];
    }

    __syncthreads();

    if(i>=1 && i<side-1 && j>=1 && j<side-1 && k>=1 && k<side-1){
        if(threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1 && threadIdx.y>=1 && threadIdx.y<IN_TILE_DIM-1 && threadIdx.z>=1 && threadIdx.z<IN_TILE_DIM-1){
            out[i*side*side + j*side + k] = c0*inp_s[threadIdx.z][threadIdx.y][threadIdx.x];
                                            + c1*inp_s[threadIdx.z][threadIdx.y][threadIdx.x-1];
                                            + c2*inp_s[threadIdx.z][threadIdx.y][threadIdx.x+1];
                                            + c3*inp_s[threadIdx.z][threadIdx.y-1][threadIdx.x];
                                            + c4*inp_s[threadIdx.z][threadIdx.y+1][threadIdx.x];
                                            + c5*inp_s[threadIdx.z-1][threadIdx.y][threadIdx.x];
                                            + c6*inp_s[threadIdx.z+1][threadIdx.y][threadIdx.x];;
                            
        }
    }

}

void tiled_stencil_host(float *inp_h, float *out_h, int side) {
    int size = side * side * side;
    int bytes = size * sizeof(float);

    
    float *inp_d, *out_d;
    cudaMalloc(&inp_d, bytes);
    cudaMalloc(&out_d, bytes);

    
    cudaMemcpy(inp_d, inp_h, bytes, cudaMemcpyHostToDevice);

    
    cudaMemset(out_d, 0, bytes);

    
    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDim((side + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (side + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (side + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    
    tiled_stencil<<<gridDim, blockDim>>>(inp_d, out_d, side);

   
    cudaDeviceSynchronize();

    
    cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost);

    
    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    const int side = 16; 
    int size = side * side * side;

    
    float *inp_h = new float[size];
    float *out_h = new float[size];

    
    for (int i = 0; i < size; i++) {
        inp_h[i] = static_cast<float>(i);
    }

    
    float h_c0 = 1.0f; 
    float h_c1 = 0.5f; 
    float h_c2 = 0.5f; 
    float h_c3 = 0.5f; 
    float h_c4 = 0.5f; 
    float h_c5 = 0.5f;
    float h_c6 = 0.5f;

    cudaMemcpyToSymbol(c0, &h_c0, sizeof(float));
    cudaMemcpyToSymbol(c1, &h_c1, sizeof(float));
    cudaMemcpyToSymbol(c2, &h_c2, sizeof(float));
    cudaMemcpyToSymbol(c3, &h_c3, sizeof(float));
    cudaMemcpyToSymbol(c4, &h_c4, sizeof(float));
    cudaMemcpyToSymbol(c5, &h_c5, sizeof(float));
    cudaMemcpyToSymbol(c6, &h_c6, sizeof(float));

    
    tiled_stencil_host(inp_h, out_h, side);

    
    cout << "Sample output (interior points: i=1, j=1, k=1 to 5):" << endl;
    for (int k = 1; k <= 5; k++) {
        int i = 1, j = 1;
        int index = i * side * side + j * side + k;
        cout << out_h[index] << " ";
    }
    cout << endl;

    
    delete[] inp_h;
    delete[] out_h;

    return 0;
}