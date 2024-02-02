
#include "gpubuff.cu.cuh"
#include "shuffler.cu.cuh"
#include "smem_debug.cuh"
#include "test_shuffler.cuh"

#include <cuda_fp16.h>
#include <driver_types.h>
#include <iostream>
#include <mma.h>

#include <cstdio>
#include <random>

using namespace e;
using namespace std;

template <class T, class U>
T *multiply(U *a, // m  x _k
            U *b, // _k x  n
            int m, int n, int _k) {
  T *buff = new T[m * n];

  View av{.m = m, .n = _k};
  View bv{.m = _k, .n = n};

  for (int i = 0; i < m * n; i++)
    buff[i] = T(0);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < _k; k++) {
        if (std::is_same<U, half>::value) {
          buff[i * n + j] += a[av(i, k)] * b[bv(k, j)];
        } else {
          buff[i * n + j] += T(a[av(i, k)]) * T(b[bv(k, j)]);
        }
      }
    }
  }
  return buff;
}

template <int SP, int DE, bool SMEM, bool SPARSE_PIPELINE>
void validate(int m, int n, int k, const int TILE_SIZE, unsigned int pad_m,
              unsigned int pad_n, ScopedBuffer<half> &sparse,
              ScopedBuffer<half> &dense) {

  auto d_output_values =
      multiply_host<half, half, MultiplyConfig<SP, DE, SMEM, SPARSE_PIPELINE>>(
          sparse.data(), dense.data(), m, n, k, true, TILE_SIZE, 10);
  HANDLE_ERROR(cudaDeviceSynchronize());
  half *output_values = new half[pad_m * pad_n];
  D2H(output_values, d_output_values, pad_m * pad_n, half);
  half *ground_truth =
      multiply<half, half>(sparse.data(), dense.data(), m, n, k);
  double r =
      result<half, half>(ground_truth, m, n, output_values, pad_m, pad_n);
  printf("error: %lf\n", r);
  if (r > 0.002) {
    int x{};
    exit(1);
  }
}

template <class T> void fill_dense(T *base, int n, int tile_size, int val) {
  int id = val;

  std::random_device rd;

  //
  // Engines
  //
  std::mt19937 e2(rd());
  // std::knuth_b e2(rd());
  // std::default_random_engine e2(rd()) ;

  //
  // Distribtuions
  //
  std::uniform_real_distribution<> dist(0, 0.5);

  if constexpr (std::is_same<T, half>::value) {
    // Convert to fp16 (half-precision)
    for (int k = 0; k < tile_size; k++) {
      for (int l = 0; l < tile_size; l++) {
        base[k * n + l] = id;
      }
    }
  } else {
    int id{};
    for (int k = 0; k < tile_size; k++) {
      for (int l = 0; l < tile_size; l++) {
        base[k * n + l] = ++id;
      }
    }
  }
}

template <class T> void fill_sparse(T *base, int n, int tile_size) {
  int nnz = 8; // (rand() % 8) + 1;
  printf("nnz: %d\n", nnz);
  for (int i = 0; i < nnz; i++) {
    int x = rand() % tile_size;
    int y = rand() % tile_size;
    base[x * n + y] = 0.1;
  }
}

template <class T> Buffer<T> generate_test_case_checkerboard(int m, int n) {
  int tile_size = 16;
  Buffer<T> buff(m * n);
  bool sparse = true;

  int id = 0;
  for (int i = 0; i < m; i += tile_size) {
    for (int j = 0; j < n; j += tile_size) {
      int offset = i * n + j;
      if (sparse) {
        fill_sparse(buff.data() + offset, n, tile_size);
      } else {
        fill_dense(buff.data() + offset, n, tile_size, 1);
      }
      sparse ^= true;
    }
  }

  return buff;
}

int main() {
  using ValueType = half;
  const char *raw_mat =
      "100000000000100000000000000000100110000000000001010100000000100000001000"
      "000000000000010000000000000000100000000000000001000000000000000000000001"
      "000000000000010000000000000010000000000000000100000010100000000000001001"
      "0000100000000001000000100000100000000000";
  //  "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";
  const char *sparse_mat = "100000000000000000000000000000000000000000000"
                           "000000000000000000000000000"
                           "000000000000000000000000000000000000000000000"
                           "000000000000000000000000000"
                           "000000000000000000000000000000000000000000000"
                           "010000000000000000000000000"
                           "0000000000001000000000000000000000000001";

  int base = 16;
  for (int m = base; m < 17;) {
    for (int n = base; n < 17;) {
      for (int k = base; k < 17;) {
        printf("m n k %d %d %d\n", m, n, k);
        constexpr int TILE_SIZE = 16;
        unsigned int pad_m = TILE_SIZE * ((m + TILE_SIZE - 1) / TILE_SIZE);
        unsigned int pad_n = TILE_SIZE * ((n + TILE_SIZE - 1) / TILE_SIZE);

        ScopedBuffer<half> sparse(generate_test_case_checkerboard<half>(m, k));
        ScopedBuffer<half> dense(k * n, 1);
        // validate<1, 1>(m, n, k, TILE_SIZE, pad_m, pad_n, sparse);
        validate<1, 2, true, true>(m, n, k, TILE_SIZE, pad_m, pad_n, sparse,
                                   dense);

        // validate<1, 8>(m, n, k, TILE_SIZE, pad_m, pad_n, sparse);
        // validate<2, 4>(m, n, k, TILE_SIZE, pad_m, pad_n, sparse);
      }
    }
  }

  return 0;
}

__global__ void get_pwned() {
  static constexpr int tile_size = 16;
  __shared__ half buff[tile_size * tile_size];
  clr_bless_async<half, ThreadDim::X>(buff, tile_size * tile_size,
                                      __float2half(2049.f));
  __syncthreads();
  debug_shared_sync0("buff", buff, tile_size, tile_size);
  atomicAdd(&buff[0], 1);
  debug_shared_sync0("buff", buff, tile_size, tile_size);
}

int _main() {
  get_pwned<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}