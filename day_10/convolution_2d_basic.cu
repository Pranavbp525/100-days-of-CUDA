#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void convolution_2d_basic_kernel(float * inp, float * filter, float * out, int filter_size, int width, int height){

    int out_col = threadIdx.x + blockIdx.x * blockDim.x;
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;

    float val = 0.0f;

    for(int filter_row = 0; filter_row < 2 * filter_size + 1; filter_row++){
        for(int filter_col = 0; filter_col < 2 * filter_size + 1; filter_col++){
            int inp_row = out_row + filter_row - filter_size;
            int inp_col = out_col + filter_col - filter_size;

            if(inp_row>=0 && inp_row<height && inp_col>=0 && inp_col<width){
                val += filter[filter_row][filter_col] * inp[inp_row*width + inp_col];
            }
        }
    }

    out[out_row][out_col] = val;

}


void convolution_2d_basic_host(float * inp_h, float * filter_h, float * out_h, int filter_size, int width, int height){

    int full_filter_size = 2 * filter_size + 1;
    int input_size = width * height * sizeof(float);
    int output_size = (width - full_filter_size + 1) * (height - full_filter_size + 1) * sizeof(float);
    int filter_size_bytes = full_filter_size * full_filter_size * sizeof(float);

    float *inp_d, *out_d, *filter_d;

    cudaMalloc((void**)&inp_d, input_size);
    cudaMalloc((void**)&out_d, output_size);
    cudaMalloc((void**)&filter_d, filter_size_bytes);

    cudaMemcpy(inp_d, inp_h, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d, filter_h, filter_size_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(4, 4, 1);
    dim3 gridDim((width - full_filter_size + 1 + blockDim.x - 1) / blockDim.x, 
                 (height - full_filter_size + 1 + blockDim.y - 1) / blockDim.y, 1);

    convolution_2d_basic_kernel<<<gridDim, blockDim>>>(inp_d, filter_d, out_d, filter_size, width, height);

    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, output_size, cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
    cudaFree(filter_d);




}


int main() {
    const int width = 16;
    const int height = 16;
    const int filter_size = 2; 
    const int full_filter_size = 2 * filter_size + 1;
    const int out_width = width - full_filter_size + 1;  
    const int out_height = height - full_filter_size + 1; 

    const int input_size = width * height;
    const int filter_size_total = full_filter_size * full_filter_size;
    const int output_size = out_width * out_height;

    float* inp_h = new float[input_size];
    float* filter_h = new float[filter_size_total];
    float* out_h = new float[output_size];

    
    for (int i = 0; i < input_size; i++) {
        inp_h[i] = static_cast<float>(i + 1);
    }

    
    for (int i = 0; i < filter_size_total; i++) {
        filter_h[i] = 1.0f / 25.0f;
    }

    convolution_2d_basic_host(inp_h, filter_h, out_h, filter_size, width, height);

    
    cout << "Input Matrix (16x16):" << endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << inp_h[i * width + j] << " ";
        }
        cout << endl;
    }

    
    cout << "\nFilter (5x5):" << endl;
    for (int i = 0; i < full_filter_size; i++) {
        for (int j = 0; j < full_filter_size; j++) {
            cout << filter_h[i * full_filter_size + j] << " ";
        }
        cout << endl;
    }

    
    cout << "\nOutput Matrix (12x12):" << endl;
    for (int i = 0; i < out_height; i++) {
        for (int j = 0; j < out_width; j++) {
            cout << out_h[i * out_width + j] << " ";
        }
        cout << endl;
    }

    delete[] inp_h;
    delete[] filter_h;
    delete[] out_h;

    return 0;
}