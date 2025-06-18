#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h> 

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}






__global__ void predict_x0_from_noise_kernel(
    float* pred_x0_out, 
    const float* x_t_in, 
    const float* predicted_noise_in,
    float alpha_bar_t,
    int n_elements) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        float sqrt_alpha_bar_t = sqrtf(alpha_bar_t);
        float sqrt_one_minus_alpha_bar_t = sqrtf(1.0f - alpha_bar_t);
        pred_x0_out[i] = (1.0f / sqrt_alpha_bar_t) * (x_t_in[i] - sqrt_one_minus_alpha_bar_t * predicted_noise_in[i]);
    }
}


__global__ void get_x_prev_and_add_noise_kernel(
    float* x_prev_out,
    const float* pred_x0_in,
    const float* x_t_in,
    curandState* curand_state,
    int t,
    const float* alphas_bar,
    const float* betas,
    int n_elements) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        float alpha_bar_t = alphas_bar[t];
        float alpha_bar_t_prev = (t > 0) ? alphas_bar[t - 1] : 1.0f;
        float beta_t = betas[t];
        
        float posterior_mean_coef1 = (sqrtf(alpha_bar_t_prev) * beta_t) / (1.0f - alpha_bar_t);
        float posterior_mean_coef2 = (sqrtf(1.0f - beta_t) * (1.0f - alpha_bar_t_prev)) / (1.0f - alpha_bar_t);
        
        float posterior_mean = posterior_mean_coef1 * pred_x0_in[i] + posterior_mean_coef2 * x_t_in[i];

        if (t > 0) {
            curandState local_state = curand_state[i];
            float noise = curand_normal(&local_state);
            curand_state[i] = local_state; 

            float posterior_variance = ((1.0f - alpha_bar_t_prev) / (1.0f - alpha_bar_t)) * beta_t;
            x_prev_out[i] = posterior_mean + sqrtf(posterior_variance) * noise;
        } else {
            x_prev_out[i] = posterior_mean;
        }
    }
}



void print_tensor(const std::string& name, const float* m, int C, int H, int W) {
    std::cout << "--- " << name << " --- (C" << C << ", H" << H << ", W" << W << ")\n";
    std::vector<float> h_m(C * H * W);
    CHECK_CUDA(cudaMemcpy(h_m.data(), m, C * H * W * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < std::min(8, H); ++i) {
        for (int j = 0; j < std::min(8, W); ++j) {
            std::cout << h_m[i * W + j] << "\t";
        }
        std::cout << (W > 8 ? "...\n" : "\n");
    }
    std::cout << (H > 8 ? "...\n" : "");
    std::cout << "-----------------------\n" << std::endl;
}


int main() {
    const int C = 3, H = 64, W = 64; 
    const int total_elements = C * H * W;
    const int num_timesteps = 1000;

    std::vector<float> h_betas(num_timesteps);
    std::vector<float> h_alphas(num_timesteps);
    std::vector<float> h_alphas_bar(num_timesteps);

    float beta_start = 0.0001f;
    float beta_end = 0.02f;
    for (int i = 0; i < num_timesteps; ++i) {
        h_betas[i] = beta_start + (float)i * (beta_end - beta_start) / (float)(num_timesteps - 1);
        h_alphas[i] = 1.0f - h_betas[i];
        h_alphas_bar[i] = (i > 0) ? h_alphas_bar[i-1] * h_alphas[i] : h_alphas[i];
    }
    
    float *d_image, *d_predicted_noise, *d_pred_x0;
    float *d_betas, *d_alphas_bar;
    curandState *d_curand_state;

    CHECK_CUDA(cudaMalloc(&d_image, total_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_predicted_noise, total_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pred_x0, total_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_betas, num_timesteps * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_alphas_bar, num_timesteps * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_curand_state, total_elements * sizeof(curandState)));

    CHECK_CUDA(cudaMemcpy(d_betas, h_betas.data(), num_timesteps * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_alphas_bar, h_alphas_bar.data(), num_timesteps * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_initial_noise(total_elements);
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for(float& v : h_initial_noise) v = dis(gen);
    CHECK_CUDA(cudaMemcpy(d_image, h_initial_noise.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "--- Full Diffusion Model Generation Loop ---" << std::endl;
    print_tensor("Initial Pure Noise (x_T)", d_image, C, H, W);

    dim3 blocks((total_elements + 255) / 256);
    dim3 threads(256);
    
    CHECK_CUDA(cudaMemcpy(d_predicted_noise, h_initial_noise.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int t = num_timesteps - 1; t >= 0; --t) {
        if (t % 100 == 0) {
            std::cout << "Denoising at timestep t = " << t << std::endl;
        }

        predict_x0_from_noise_kernel<<<blocks, threads>>>(d_pred_x0, d_image, d_predicted_noise, h_alphas_bar[t], total_elements);
        get_x_prev_and_add_noise_kernel<<<blocks, threads>>>(d_image, d_pred_x0, d_image, d_curand_state, t, d_alphas_bar, d_betas, total_elements);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << "\nGeneration complete." << std::endl;
    print_tensor("Final Generated Image (x_0)", d_image, C, H, W);

    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_image)); CHECK_CUDA(cudaFree(d_predicted_noise)); CHECK_CUDA(cudaFree(d_pred_x0));
    CHECK_CUDA(cudaFree(d_betas)); CHECK_CUDA(cudaFree(d_alphas_bar)); CHECK_CUDA(cudaFree(d_curand_state));

    return 0;
}
