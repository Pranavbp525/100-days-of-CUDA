#include<iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

__global__ void blur_greyscale(unsigned char * inp, unsigned chat * out, int blur_size, int w, int h){

  int row = blockIdx.x*blockDim.x + threadIdx.x;
  int col = blockIdx.y*blockDim.y + threadIdx.y;

  if (row < h && col < w){

    int sum = 0;
    int num = 0;
    int curr_pixel_one_d_index_out = row * w + col;

    for(i = -blur_size; i<=blur_size; ++i){
      for(j = -blur_size; j<=blur_size; ++j){

        int curr_row = row + i;
        int curr_col = col + i;

        if (curr_row>=0 && curr_row<h && curr_col>=0 && curr_col<w){

          int curr_pixel_one_d_index = curr_row * w + curr_col;
          sum = sum + inp[curr_pixel_one_d_index];
          num = num + 1;

        }
      }
    }


    out[curr_pixel_one_d_index_out] = (unsigned char)(sum/num);

  }

}


void blur_greyscale(unsigned char * inp_h, unsigned char * out_h, int blur_size, int w, int h){

  unsigned char * inp_d;
  unsigned char * out_d;

  int size = w*h*sizeof(unsigned char);

  cudaMalloc((**void)inp_d, size);
  cudaMalloc((**void)out_d, size);

  cudaMemcpy(inp_d, out_d, size, cudaMemcpyHostToDevice);

  dim3 blockDim(256, 256, 1);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

  blur_greyscale<<<gridDim, blockDim>>>(inp_d, out_d, blur_size, w, h);

  cudaDeviceSynchronize();

  cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

  cudaFree(inp_d);
  cudaFree(out_d);

}





int main() {

  std::string input_path = "/content/drive/MyDrive/CUDA/Day_2/grey_output.jpg"; // From Day 8
  Mat img = imread(input_path, IMREAD_GRAYSCALE);
  if (img.empty()) {
      cerr << "Error: Could not load image" << endl;
      return -1;
  }

  int w = img.cols;
  int h = img.rows;
  const int blur_size = 1;
  unsigned char* inp_h = img.data;
  unsigned char* out_h = new unsigned char[w * h];

  blur_greyscale_host(inp_h, out_h, blur_size, w, h);

  Mat blurred_img(h, w, CV_8UC1, out_h);
  std::string output_path = "/content/drive/MyDrive/CUDA/Day_3/blurred_output.jpg";
  imwrite(output_path, blurred_img);

  delete[] out_h;
  cout << "Blurred image saved to " << output_path << endl;
  return 0;

}
