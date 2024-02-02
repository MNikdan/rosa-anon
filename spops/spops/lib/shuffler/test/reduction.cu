#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <memory>
#include <iostream>
#include <cassert>

#include "utility.cuh"
#include "gpu_timer.cuh"

using T = int;

__global__ void init(T *arr) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  arr[id] = id + 1;
}

__global__ void reduceAtomicGlobal(const T *input, T *result, int elements) {
  unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= elements) {
    return;
  }
  atomicAdd(result, input[id]);
}

__global__ void reduceAtomicShared(const T *input, T *result, int elements) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= elements) {
    return;
  }
  __shared__ T value;
  value = T(0);

  __syncthreads();
  atomicAdd(&value, input[id]);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(result, value);
  }
}

__global__ void atomicShared(const T *input, T *result, int elements) {
  extern __shared__ T data[];
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= elements) {
    return;
  }

  data[threadIdx.x] = input[id];

  __syncthreads();

  // Problem: Thread diveregence: not all threads are doing the same amount of work.
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (threadIdx.x % (2 * s) == 0) {
      data[threadIdx.x] += data[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, data[0]);
  }
}

__global__ void shareAntiDivergenceAtomic(const T *input, T *result, int elements) {
  extern __shared__ T data[];
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= elements) {
    return;
  }

  data[threadIdx.x] = input[id];

  __syncthreads();

  for (int s = 1; s < blockDim.x; s *= 2) {
    int lid = 2 * s * threadIdx.x;
    // Has bank conflicts
    if (lid < blockDim.x) {
      data[lid] += data[lid + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, data[0]);
  }
}

__global__ void sharedAntiConflicts(const T *input, T *result, int elements) {
  extern __shared__ T data[];
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= elements) {
    return;
  }

  data[threadIdx.x] = input[id];

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s)
      data[threadIdx.x] += data[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, data[0]);
  }
}

__global__ void reduceSharedDualLoad(const T *input, T *result, int elements) {
  extern __shared__ T data[];
  int id = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= elements) {
    return;
  }

  data[threadIdx.x] = input[id] + input[id + blockDim.x];

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s) {
      data[threadIdx.x] += data[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, data[0]);
  }
}


__global__ void sharedUnrollLast(const T *input, T *result, int elements) {
  extern __shared__ volatile T data0[];
  int id = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= elements) {
    return;
  }

  data0[threadIdx.x] = input[id] + input[id + blockDim.x];

  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s /= 2) {
    if (threadIdx.x < s) {
      data0[threadIdx.x] += data0[threadIdx.x + s];
    }
    __syncthreads();
  }

  T x{data0[threadIdx.x]};

  if (threadIdx.x < 32) {
    data0[threadIdx.x] += data0[threadIdx.x + 32]; __syncwarp(); data0[threadIdx.x] = x; __syncwarp();
    data0[threadIdx.x] += data0[threadIdx.x + 16]; __syncwarp(); data0[threadIdx.x] = x; __syncwarp();
    data0[threadIdx.x] += data0[threadIdx.x +  8]; __syncwarp(); data0[threadIdx.x] = x; __syncwarp();
    data0[threadIdx.x] += data0[threadIdx.x +  4]; __syncwarp(); data0[threadIdx.x] = x; __syncwarp();
    data0[threadIdx.x] += data0[threadIdx.x +  2]; __syncwarp(); data0[threadIdx.x] = x; __syncwarp();
  }

  if (threadIdx.x == 0) {
    atomicAdd(result, data0[0] + data0[1]);
  }
}

int main() {
  int N = 1 << 5;
  int thread_count = 4;
  int ground_truth = N * (N + 1) / 2;

  T *d_array;

  CUMALLOC0(d_array, N, T);

  init<<<(N + thread_count - 1) / thread_count, thread_count>>>(d_array);
  cudaDeviceSynchronize();

  int *d_result;
  T *result;
  cudaMalloc((void **) &d_result, sizeof(T));

  {
    Timer t(1, "reduceAtomicGlobal", Timer::Type::GPU);
    cudaMemset(d_result, 0, sizeof(T));
    cudaDeviceSynchronize();
    t.start();
    reduceAtomicGlobal<<<(N + thread_count - 1) / thread_count, thread_count>>>(d_array, d_result, N);
    cudaDeviceSynchronize();
    t.end();
    t.finalize_benchmark();
    result = new T[1];
    cudaMemcpy(result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    assert(*result == ground_truth);
  }

  {
    Timer t(1, "reduceAtomicShared", Timer::Type::GPU);
    cudaMemset(d_result, 0, sizeof(T));
    cudaDeviceSynchronize();
    t.start();
    reduceAtomicShared<<<(N + thread_count - 1) / thread_count, thread_count>>>(d_array, d_result, N);
    cudaDeviceSynchronize();
    t.end();
    t.finalize_benchmark();
    result = new T[1];
    cudaMemcpy(result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    assert(*result == ground_truth);
  }

  {
    Timer t(1, "atomicShared", Timer::Type::GPU);
    cudaMemset(d_result, 0, sizeof(T));
    cudaDeviceSynchronize();
    t.start();
    atomicShared<<<(N + thread_count - 1) / thread_count, thread_count, sizeof(T) * thread_count>>>(d_array,
                                                                                                    d_result,
                                                                                                    N);
    cudaDeviceSynchronize();
    t.end();
    t.finalize_benchmark();
    result = new T[1];
    cudaMemcpy(result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    assert(*result == ground_truth);
  }

  {
    Timer t(1, "shareAntiDivergenceAtomic", Timer::Type::GPU);
    cudaMemset(d_result, 0, sizeof(T));
    cudaDeviceSynchronize();
    t.start();
    shareAntiDivergenceAtomic<<<(N + thread_count - 1) / thread_count, thread_count, sizeof(T)
        * thread_count>>>(d_array,
                          d_result,
                          N);
    cudaDeviceSynchronize();
    t.end();
    t.finalize_benchmark();
    result = new T[1];
    cudaMemcpy(result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    assert(*result == ground_truth);
  }

  {
    Timer t(1, "sharedAntiConflicts", Timer::Type::GPU);
    cudaMemset(d_result, 0, sizeof(T));
    cudaDeviceSynchronize();
    t.start();
    sharedAntiConflicts<<<(N + thread_count - 1) / thread_count, thread_count, sizeof(T)
        * thread_count>>>(d_array,
                          d_result,
                          N);
    cudaDeviceSynchronize();
    t.end();
    t.finalize_benchmark();
    result = new T[1];
    cudaMemcpy(result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    assert(*result == ground_truth);
  }

  {
    Timer t(1, "reduceSharedDualLoad", Timer::Type::GPU);
    cudaMemset(d_result, 0, sizeof(T));
    cudaDeviceSynchronize();
    t.start();
    reduceSharedDualLoad<<<((N + thread_count - 1) / thread_count) / 2, thread_count, sizeof(T)
        * thread_count>>>(d_array,
                          d_result,
                          N);
    cudaDeviceSynchronize();
    t.end();
    t.finalize_benchmark();
    result = new T[1];
    cudaMemcpy(result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    assert(*result == ground_truth);
  }

  {
    Timer t(1, "sharedUnrollLast", Timer::Type::GPU);
    cudaMemset(d_result, 0, sizeof(T));
    cudaDeviceSynchronize();
    t.start();
    sharedUnrollLast<<<((N + thread_count - 1) / thread_count) / 2, thread_count, sizeof(T) * thread_count>>>(d_array, d_result, N);
    cudaDeviceSynchronize();
    t.end();
    t.finalize_benchmark();
    result = new T[1];
    cudaMemcpy(result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << *result << ' ' << ground_truth << std::endl;
    assert(*result == ground_truth);
  }

  return 0;
}



















