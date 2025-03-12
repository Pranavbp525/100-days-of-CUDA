#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

__global__ void color2greyKernel(unsigned char* inp, unsigned char* out, int height, int width) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < height && col < width) {
        int one_d_grey_index = row * width + col;
        int one_d_color_index = one_d_grey_index * 3;

        float r = inp[one_d_color_index];
        float g = inp[one_d_color_index + 1];
        float b = inp[one_d_color_index + 2];
        out[one_d_grey_index] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

void color2greyHost(unsigned char* inp_h, unsigned char* out_h, int height, int width) {
    int inp_size = 3 * height * width * sizeof(unsigned char);
    int out_size = height * width * sizeof(unsigned char);

    unsigned char *inp_d, *out_d;

    cudaMalloc((void**)&inp_d, inp_size);
    cudaMalloc((void**)&out_d, out_size);

    cudaMemcpy(inp_d, inp_h, inp_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16, 1); // 256 threads per block
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);

    color2greyKernel<<<dimGrid, dimBlock>>>(inp_d, out_d, height, width);

    cudaDeviceSynchronize();

    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);

    cudaFree(inp_d);
    cudaFree(out_d);
}

int main() {
    // Load image from Google Drive
    std::string input_path = "/content/drive/MyDrive/CUDA/Day_2/input.jpg";
    Mat img = imread(input_path, IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << input_path << std::endl;
        return -1;
    }

    int height = img.rows;
    int width = img.cols;
    unsigned char* inp_h = img.data; // OpenCV uses BGR order
    unsigned char* out_h = new unsigned char[height * width];

    // Convert to grayscale
    color2greyHost(inp_h, out_h, height, width);

    // Save output image
    Mat grey_img(height, width, CV_8UC1, out_h);
    std::string output_path = "/content/drive/MyDrive/CUDA/Day_2/grey_output.jpg";
    imwrite(output_path, grey_img);

    delete[] out_h;
    std::cout << "Grayscale image saved to " << output_path << std::endl;

    return 0;
}