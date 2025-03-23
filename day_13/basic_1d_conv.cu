#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void convolution_1d_basic_kernel(float* input, float* filter, float* output, int filter_size, int input_size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int filter_radius = filter_size / 2;
    int output_size = input_size - filter_size + 1;

    if (i < output_size) {
        float val = 0.0f;
        for (int k = 0; k < filter_size; ++k) {
            int input_idx = i + k - filter_radius;
            if (input_idx >= 0 && input_idx < input_size) {
                val += filter[k] * input[input_idx];
            }
        }
        output[i] = val;
    }
}

void convolution_1d_basic_host(float* input_h, float* filter_h, float* output_h, int filter_size, int input_size) {
    int output_size = input_size - filter_size + 1;
    int input_bytes = input_size * sizeof(float);
    int filter_bytes = filter_size * sizeof(float);
    int output_bytes = output_size * sizeof(float);

    float *input_d, *filter_d, *output_d;

    cudaMalloc((void**)&input_d, input_bytes);
    cudaMalloc((void**)&filter_d, filter_bytes);
    cudaMalloc((void**)&output_d, output_bytes);

    cudaMemcpy(input_d, input_h, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d, filter_h, filter_bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (output_size + block_size - 1) / block_size;

    convolution_1d_basic_kernel<<<grid_size, block_size>>>(input_d, filter_d, output_d, filter_size, input_size);

    cudaDeviceSynchronize();

    cudaMemcpy(output_h, output_d, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(filter_d);
    cudaFree(output_d);
}

int main() {
    const int input_size = 16;
    const int filter_size = 5;
    const int output_size = input_size - filter_size + 1;

    float* input_h = new float[input_size];
    float* filter_h = new float[filter_size];
    float* output_h = new float[output_size];

    for (int i = 0; i < input_size; i++) {
        input_h[i] = static_cast<float>(i + 1);
    }

    for (int i = 0; i < filter_size; i++) {
        filter_h[i] = 1.0f / filter_size;
    }

    convolution_1d_basic_host(input_h, filter_h, output_h, filter_size, input_size);

    cout << "Input Array (16 elements):" << endl;
    for (int i = 0; i < input_size; i++) {
        cout << input_h[i] << " ";
    }
    cout << endl;

    cout << "\nFilter (5 elements):" << endl;
    for (int i = 0; i < filter_size; i++) {
        cout << filter_h[i] << " ";
    }
    cout << endl;

    cout << "\nOutput Array (12 elements):" << endl;
    for (int i = 0; i < output_size; i++) {
        cout << output_h[i] << " ";
    }
    cout << endl;

    delete[] input_h;
    delete[] filter_h;
    delete[] output_h;

    return 0;
}