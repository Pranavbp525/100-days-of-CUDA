#include <iostream>
#include <cuda_runtime.h>
#include <string>

using namespace std;


#define NUM_BINS 7  // 26 letters grouped by 4, ceil(26/4) = 7 bins


__global__ void simple_histogram(char *data, int length, int *hist) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < length) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(hist[pos / 4]), 1);  
        }
    }
}


void simple_histogram_host(char *data_h, int length, int *hist_h) {
    
    int inp_size = sizeof(char) * length;
    int out_size = sizeof(int) * NUM_BINS; 

    char *data_d;
    int *hist_d;

    
    cudaMalloc((void**)&data_d, inp_size);
    cudaMalloc((void**)&hist_d, out_size);

    
    cudaMemcpy(data_d, data_h, inp_size, cudaMemcpyHostToDevice);

    
    cudaMemset(hist_d, 0, out_size);

    int block_size = 256; 
    int grid_size = (length + block_size - 1) / block_size;

    simple_histogram<<<grid_size, block_size>>>(data_d, length, hist_d);

    cudaDeviceSynchronize();

    cudaMemcpy(hist_h, hist_d, out_size, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
    cudaFree(hist_d);
}

int main() {
    string input_text = "one hundred days of cuda challenge";
    int length = input_text.length();

    char *data_h = new char[length];
    int *hist_h = new int[NUM_BINS]();

    memcpy(data_h, input_text.c_str(), length);

    simple_histogram_host(data_h, length, hist_h);

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