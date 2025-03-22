#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define FILTER_RADIUS 2
#define FILTER_DIAMETER (2 * FILTER_RADIUS + 1)
#define BLOCK_SIZE 16

__constant__ float d_Filter[FILTER_DIAMETER * FILTER_DIAMETER];

__global__ void convolution2D_tiled_kernel(const float* __restrict__ d_input,
                                           float* __restrict__ d_output,
                                           int width, int height)
{
    const int OUT_TILE_SIZE = BLOCK_SIZE;
    int out_x = blockIdx.x * OUT_TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * OUT_TILE_SIZE + threadIdx.y;

    __shared__ float s_tile[BLOCK_SIZE + 2 * FILTER_RADIUS][BLOCK_SIZE + 2 * FILTER_RADIUS];

    int halo_x = (blockIdx.x * OUT_TILE_SIZE) + threadIdx.x - FILTER_RADIUS;
    int halo_y = (blockIdx.y * OUT_TILE_SIZE) + threadIdx.y - FILTER_RADIUS;

    if (halo_x < 0) halo_x = 0;
    if (halo_x >= width) halo_x = width - 1;
    if (halo_y < 0) halo_y = 0;
    if (halo_y >= height) halo_y = height - 1;

    s_tile[threadIdx.y][threadIdx.x] = d_input[halo_y * width + halo_x];

    __syncthreads();

    if (out_x < width && out_y < height) 
    {
        if (out_x >= FILTER_RADIUS && out_x < (width - FILTER_RADIUS) &&
            out_y >= FILTER_RADIUS && out_y < (height - FILTER_RADIUS))
        {
            float sum = 0.0f;

            for (int ky = -FILTER_RADIUS; ky <= FILTER_RADIUS; ky++) 
            {
                for (int kx = -FILTER_RADIUS; kx <= FILTER_RADIUS; kx++) 
                {
                    float pixel = s_tile[threadIdx.y + ky + FILTER_RADIUS][threadIdx.x + kx + FILTER_RADIUS];
                    float coeff = d_Filter[(ky + FILTER_RADIUS) * FILTER_DIAMETER + (kx + FILTER_RADIUS)];
                    sum += pixel * coeff;
                }
            }

            int out_index = (out_y - FILTER_RADIUS) * (width - 2 * FILTER_RADIUS) + (out_x - FILTER_RADIUS);
            d_output[out_index] = sum;
        }
    }
}

void convolution2D_tiled_host(const float* h_input,
                              const float* h_filter,
                              float* h_output,
                              int width,
                              int height)
{
    int input_size = width * height;
    size_t input_bytes = input_size * sizeof(float);

    int out_width  = width  - 2 * FILTER_RADIUS;
    int out_height = height - 2 * FILTER_RADIUS;
    size_t output_bytes = out_width * out_height * sizeof(float);

    cudaMemcpyToSymbol(d_Filter, h_filter, FILTER_DIAMETER * FILTER_DIAMETER * sizeof(float));

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc((void**)&d_input, input_bytes);
    cudaMalloc((void**)&d_output, output_bytes);

    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolution2D_tiled_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    const int width = 16;
    const int height = 16;

    int out_width  = width  - 2 * FILTER_RADIUS;
    int out_height = height - 2 * FILTER_RADIUS;

    float* h_input  = new float[width * height];
    float* h_filter = new float[FILTER_DIAMETER * FILTER_DIAMETER];
    float* h_output = new float[out_width * out_height];

    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    for (int i = 0; i < FILTER_DIAMETER * FILTER_DIAMETER; i++) {
        h_filter[i] = 1.0f / 25.0f;
    }

    convolution2D_tiled_host(h_input, h_filter, h_output, width, height);

    cout << "Input Matrix:\n";
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            cout << h_input[r * width + c] << " ";
        }
        cout << "\n";
    }

    cout << "\nFilter:\n";
    for (int r = 0; r < FILTER_DIAMETER; r++) {
        for (int c = 0; c < FILTER_DIAMETER; c++) {
            cout << h_filter[r * FILTER_DIAMETER + c] << " ";
        }
        cout << "\n";
    }

    cout << "\nOutput Matrix:\n";
    for (int r = 0; r < out_height; r++) {
        for (int c = 0; c < out_width; c++) {
            cout << h_output[r * out_width + c] << " ";
        }
        cout << "\n";
    }

    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;

    return 0;
}
