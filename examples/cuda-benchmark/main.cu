#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void vector_scale(const float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i] * 2.0f;
}

int main() {
  constexpr int kElements = 1 << 20;
  std::vector<float> input(kElements, 1.0f);
  std::vector<float> output(kElements, 0.0f);
  float *d_input = nullptr;
  float *d_output = nullptr;
  cudaError_t status = cudaSuccess;
  status = cudaMalloc(&d_input, kElements * sizeof(float));
  if (status == cudaSuccess) status = cudaMalloc(&d_output, kElements * sizeof(float));
  if (status == cudaSuccess) status = cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) vector_scale<<<(kElements + 255) / 256, 256>>>(d_input, d_output, kElements);
  if (status == cudaSuccess) status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA benchmark failed: %s\n", cudaGetErrorString(status));
    return 1;
  }
  std::printf("scaled_elements=%zu checksum=%.1f\n", output.size(), output.back());
  return 0;
}
