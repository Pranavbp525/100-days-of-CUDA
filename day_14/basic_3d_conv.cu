#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void convolution_3d_basic_kernel(float *inp, float *filter, float *out, int filter_size, int width, int height, int depth) {
    
    int out_x = threadIdx.x + blockIdx.x * blockDim.x;
    int out_y = threadIdx.y + blockIdx.y * blockDim.y;
    int out_z = threadIdx.z + blockIdx.z * blockDim.z;

    int full_filter_size = 2 * filter_size + 1;
    int out_width = width - full_filter_size + 1;
    int out_height = height - full_filter_size + 1;
    int out_depth = depth - full_filter_size + 1;

    
    if (out_x >= out_width || out_y >= out_height || out_z >= out_depth)
        return;

    float val = 0.0f;
    
    for (int filter_z = 0; filter_z < full_filter_size; filter_z++) {
        for (int filter_y = 0; filter_y < full_filter_size; filter_y++) {
            for (int filter_x = 0; filter_x < full_filter_size; filter_x++) {
                int inp_x = out_x + filter_x - filter_size;
                int inp_y = out_y + filter_y - filter_size;
                int inp_z = out_z + filter_z - filter_size;
                
                if (inp_x >= 0 && inp_x < width &&
                    inp_y >= 0 && inp_y < height &&
                    inp_z >= 0 && inp_z < depth) {
                    
                    int inp_index = inp_z * width * height + inp_y * width + inp_x;
                    int filter_index = filter_z * full_filter_size * full_filter_size + filter_y * full_filter_size + filter_x;
                    val += filter[filter_index] * inp[inp_index];
                }
            }
        }
    }
    
    int out_index = out_z * out_width * out_height + out_y * out_width + out_x;
    out[out_index] = val;
}

void convolution_3d_basic_host(float *inp_h, float *filter_h, float *out_h, int filter_size, int width, int height, int depth) {
    int full_filter_size = 2 * filter_size + 1;
    int out_width = width - full_filter_size + 1;
    int out_height = height - full_filter_size + 1;
    int out_depth = depth - full_filter_size + 1;

    size_t input_size = width * height * depth * sizeof(float);
    size_t output_size = out_width * out_height * out_depth * sizeof(float);
    size_t filter_size_bytes = full_filter_size * full_filter_size * full_filter_size * sizeof(float);

    float *inp_d, *out_d, *filter_d;
    cudaMalloc((void**)&inp_d, input_size);
    cudaMalloc((void**)&out_d, output_size);
    cudaMalloc((void**)&filter_d, filter_size_bytes);

    cudaMemcpy(inp_d, inp_h, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d, filter_h, filter_size_bytes, cudaMemcpyHostToDevice);

    
    dim3 blockDim(4, 4, 4);
    dim3 gridDim((out_width + blockDim.x - 1) / blockDim.x,
                 (out_height + blockDim.y - 1) / blockDim.y,
                 (out_depth + blockDim.z - 1) / blockDim.z);

    convolution_3d_basic_kernel<<<gridDim, blockDim>>>(inp_d, filter_d, out_d, filter_size, width, height, depth);
    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, output_size, cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
    cudaFree(filter_d);
}

int main() {
    
    const int width = 8;
    const int height = 8;
    const int depth = 8;
    const int filter_size = 1; 
    const int full_filter_size = 2 * filter_size + 1;
    const int out_width = width - full_filter_size + 1;
    const int out_height = height - full_filter_size + 1;
    const int out_depth = depth - full_filter_size + 1;

    int input_size = width * height * depth;
    int filter_size_total = full_filter_size * full_filter_size * full_filter_size;
    int output_size = out_width * out_height * out_depth;

    
    float* inp_h = new float[input_size];
    float* filter_h = new float[filter_size_total];
    float* out_h = new float[output_size];

    
    for (int i = 0; i < input_size; i++) {
        inp_h[i] = static_cast<float>(i + 1);
    }

    
    for (int i = 0; i < filter_size_total; i++) {
        filter_h[i] = 1.0f / (filter_size_total);
    }

    convolution_3d_basic_host(inp_h, filter_h, out_h, filter_size, width, height, depth);

    
    cout << "Output Volume (" << out_depth << "x" << out_height << "x" << out_width << "):" << endl;
    for (int z = 0; z < out_depth; z++) {
        cout << "Slice " << z << ":" << endl;
        for (int y = 0; y < out_height; y++) {
            for (int x = 0; x < out_width; x++) {
                int idx = z * out_width * out_height + y * out_width + x;
                cout << out_h[idx] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    delete[] inp_h;
    delete[] filter_h;
    delete[] out_h;

    return 0;
}
