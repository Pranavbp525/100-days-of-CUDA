#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define NUM_BINS 7

__global__ void aggregated_hist_kernel(char *data, int length, int *hist) {
    __shared__ int hist_s[NUM_BINS];
    
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        hist_s[bin] = 0;
    }
    __syncthreads();

    int prevBin = -1;
    int accumulator = 0;

    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    for (int i = tid; i < length; i += blockDim.x * gridDim.x) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            int bin = pos / 4;
            if (bin == prevBin) {
                ++accumulator;
            } else {
                if (accumulator > 0) {
                    atomicAdd(&(hist_s[prevBin]), accumulator);
                }
                accumulator = 1;
                prevBin = bin;
            }
        }
    }
    if (accumulator > 0) {
        atomicAdd(&(hist_s[prevBin]), accumulator);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int binVal = hist_s[bin];
        if (binVal > 0) {
            atomicAdd(&(hist[bin]), binVal);
        }
    }
}

void aggregated_hist_host(char *data_h, int length, int *hist_h) {
    int inp_size = sizeof(char) * length;
    int hist_size = sizeof(int) * NUM_BINS;

    char *data_d;
    int *hist_d;

    cudaMalloc((void**)&data_d, inp_size);
    cudaMalloc((void**)&hist_d, hist_size);

    cudaMemcpy(data_d, data_h, inp_size, cudaMemcpyHostToDevice);
    cudaMemset(hist_d, 0, hist_size);

    int block_size = 1024;
    int grid_size = 1;
    if (length < block_size) {
        block_size = length;
    }

    aggregated_hist_kernel<<<grid_size, block_size>>>(data_d, length, hist_d);

    cudaDeviceSynchronize();

    cudaMemcpy(hist_h, hist_d, hist_size, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
    cudaFree(hist_d);
}

int main() {
    string input_text = "one hundred days of cuda challenge";
    int length = input_text.length();

    char *data_h = new char[length];
    int *hist_h = new int[NUM_BINS]();  

    memcpy(data_h, input_text.c_str(), length);

    aggregated_hist_host(data_h, length, hist_h);

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