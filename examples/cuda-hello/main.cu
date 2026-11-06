#include <cuda_runtime.h>
#include <cstdio>

__global__ void hello_kernel(int *output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0] = 42;
  }
}

int main() {
  int *device_value = nullptr;
  int host_value = 0;
  cudaError_t status = cudaMalloc(&device_value, sizeof(int));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
    return 1;
  }
  hello_kernel<<<1, 32>>>(device_value);
  status = cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_value);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(status));
    return 1;
  }
  std::printf("GPU result: %d\n", host_value);
  return host_value == 42 ? 0 : 1;
}
