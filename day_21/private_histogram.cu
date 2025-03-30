#include <iostream>
#include <cuda_runtime.h>
#include <string>

using namespace std;


#define NUM_BINS 7  


__global__ void private_histogram(char *data, int length, int *hist) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(hist[blockIdx.x*NUM_BINS + pos / 4]), 1);  
        }
    }

    if(blockIdx.x>0){
        __syncthreads();
        for(int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
            int binVal = hist[blockIdx.x*NUM_BINS + bin];
            if(binVal>0){
                atomicAdd(&(hist[bin]), binVal);
            }
        }
    }
}


void private_histogram_host(char *data_h, int length, int *hist_h) {
    int inp_size = sizeof(char) * length;
    int block_size = 256; 
    int grid_size = (length + block_size - 1) / block_size; 
    int hist_size = sizeof(int) * NUM_BINS * grid_size; 

    char *data_d;
    int *hist_d;

    cudaMalloc((void**)&data_d, inp_size);
    cudaMalloc((void**)&hist_d, hist_size);

    cudaMemcpy(data_d, data_h, inp_size, cudaMemcpyHostToDevice);
    cudaMemset(hist_d, 0, hist_size); 
    private_histogram<<<grid_size, block_size>>>(data_d, length, hist_d);

    cudaDeviceSynchronize();

    cudaMemcpy(hist_h, hist_d, sizeof(int) * NUM_BINS, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
    cudaFree(hist_d);
}

int main() {
    string input_text = "one hundred days of cuda challenge";
    int length = input_text.length();

    char *data_h = new char[length];
    int *hist_h = new int[NUM_BINS](); 

    memcpy(data_h, input_text.c_str(), length);

    private_histogram_host(data_h, length, hist_h);

    cout << "Histogram of lowercase letters (grouped by 4):" << endl;
    for (int i = 0; i < NUM_BINS; i++) {
        char start_letter = 'a' + i * 4;
        char end_letter = (i == NUM_BINS - 1) ? 'z' : start_letter + 3;
        cout << "Bin " << i << " (" << start_letter << "-" << end_letter << "): " << hist_h[i] << endl;
    }

    delete[] data_h;
    delete[] hist_h;

    return 0;
}