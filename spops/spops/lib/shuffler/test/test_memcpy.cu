#include "common.cuh"
#include "cuda_fp16.h"
#include "gpubuff.cu.cuh"
#include <vector>

#include <cstdio>
#include <smem_debug.cuh>

using namespace std;

__global__ void test_memcpy64(half *input, half *output) {
  __shared__ half s_data[16 * (16 * 16)];

  memcpy2d_strided_async_2d_256_warp<half, ThreadDim::X>(
    s_data + (get_thread_id<ThreadDim::YZ>()) * 16 * 16,
    input, 16, 16,
    0, 0, 16, 16);

  __syncthreads();

  debug_shared_sync0("smem", s_data, 64 * 4, 16);
}

int main() {
  vector<half> d(16 * 16);

  for (int i = 0, id = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      d[16 * i + j] = id++;
    }
  }

  e::GPUBuff<half> buff{}, output;
  output.alloc(64 * 64);
  buff.from_vector(d);

  test_memcpy64<<<1, dim3(32, 4, 4)>>>(buff.d_data, output.d_data);

  cudaDeviceSynchronize();

  return 0;
}