#include <iostream>
#include <cstring>
#include <cassert>
#include <vector>

#include <cuda_fp16.h>
#include <cub/device/device_scan.cuh>
#include "mask_algebra.cuh"
#include "shuffler.cu.cuh"
#include "gpubuff.cu.cuh"

using namespace e;

template<class T>
uint64_t mask_from_arr(T *data, int r, int c, T *a, int &nnz) {
  uint64_t mask{};
  for (uint64_t i = 0; i < r; i++) {
    for (uint64_t j = 0; j < c; j++) {
      if (*data) {
        mask |= (i | (j << 4ull)) << (nnz * 8ull);
        a[nnz++] = *data;
      }
      data++;
    }
  }
  return mask;
}

template<class T>
void arr_from_mask(T *data, T *a, int nnz, uint64_t mask) {
  for (int i = 0; i < 256; i++)
    data[i] = 0;
  for (uint64_t i = 0; i < nnz; i++) {
    uint64_t r = mask & 0b1111ull;
    uint64_t c = (mask >> 4ull) & 0b1111ull;
    data[r * 16 + c] = a[i];
    mask >>= 8;
  }
}

void test_transpose() {
  int a[4], a_t[4];
  int arr[16 * 16];
  std::memset(arr, 0, sizeof(arr));
  arr[1] = 1;
  arr[15] = 2;
  arr[16 * 16 - 16] = 3;
  arr[16 * 16 - 1] = 4;
  int nnz{};

  uint64_t mask = mask_from_arr(arr, 16, 16, a, nnz);

  auto transposed_mask = e::transpose_col_major8<int>(a, a_t, mask, nnz, 16);

  assert(a_t[0] == 3);
  assert(a_t[1] == 1);
  assert(a_t[2] == 2);
  assert(a_t[3] == 4);

  assert(transposed_mask == 4279173616ull);
}

void test_mask() {
  auto low = lo8(0b00001111);
  assert(low == 0b1111);
  auto hi = hi8(0b11110000);
  assert(hi == 0b1111);
}

void test_matmul() {
  using ValueType = half;
  unsigned int m = 16;
  unsigned int k = 16;
  unsigned int n = 16;
  constexpr int TILE_SIZE = 16;
  unsigned int pad_m = TILE_SIZE * ((m + TILE_SIZE - 1) / TILE_SIZE);
  unsigned int pad_n = TILE_SIZE * ((n + TILE_SIZE - 1) / TILE_SIZE);

  ValueType *sparse = new ValueType[m * k];
  ValueType *dense = new ValueType[k * n];

  for (int i = 0; i < k * n; i++)
    dense[i] = 1;
  for (int i = 0; i < 16 * 16; i++)
    sparse[i] = 0;

  for (int i = 1; i < 16; i += 2) {
    sparse[16 * i + i] = 1;
  }

  auto d_output_values = multiply_host<half, half, MultiplyConfig<1, 8>>(
      sparse,
      dense,
      m,
      n,
      k,
      false,
      TILE_SIZE,
      6);
  cudaDeviceSynchronize();

  GPUBuff<half> dc{.d_data = d_output_values, .size=pad_m * pad_n};
  std::cout << std::endl;
  dc.print(pad_m, pad_n);
}

void test_sparse_post_process() {
  GPUBuff<unsigned int> row_offsets;
  GPUBuff<unsigned int> col_ids;
  GPUBuff<unsigned int> nnz;
  GPUBuff<unsigned int> sparse_count;

  row_offsets.from_vector(std::vector<unsigned int>{0, 6, 12});
  nnz.from_vector(std::vector<unsigned int>{1, 8, 16, 32, 7, 2,
                                            1, 1, 1, 32, 32, 32});
  col_ids.from_vector(std::vector<unsigned int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  sparse_count.alloc(3); // Don't care about this one.

  d_sparse_post_process<unsigned int><<<2, 1>>>(
      2,
      row_offsets.d_data,
      nnz.d_data,
      col_ids.d_data,
      sparse_count.d_data);
  HANDLE_ERROR(cudaDeviceSynchronize());

  unsigned int *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  GPUBuff<unsigned int> b;
  b.alloc(5);
  cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                sparse_count.d_data,
                                b.d_data,
                                3);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                sparse_count.d_data,
                                b.d_data,
                                3);

  HANDLE_ERROR(cudaDeviceSynchronize());

  b.print();
  std::cout << std::endl;

  col_ids.data = nullptr;

  // col_ids.print();
  sparse_count.print();
}

void test_sparse() {
  auto id = [](int r, int c) { return 16 * r + c; };
  std::vector<half> m(16 * 16);
  for (int i = 0; i < 8; i++) {
    m[id(2 * i, 2 * i)] = i + 1;
  }
  TileFormat<half, uint64_t, int> tf;
  tf.template from_dense<half>(16, 16, m.data(), 16, false);

  std::vector<SparseTile> sparse_tile(1);
  D2H(sparse_tile.data(), tf.d_sparse_tiles, 1, SparseTile);

  std::cout << sparse_tile[0].nnz << std::endl;

  for (uint64_t i = 0; i < 64; i += 8) {
    for (uint64_t j = 8; j > 0ull; j--) {
      std::cout << ((sparse_tile[0].coo & (1ull << (i + j - 1))) > 0ull);
      if (j == 5) std::cout << ' ';
    }
    std::cout << "    ";
  }
  std::cout << std::endl;
  std::cout << sparse_tile[0].coo << std::endl;
  for (int i = 0; i < 8; i++) {
    std::cout << __half2float(reinterpret_cast<half*>(&sparse_tile[0].values)[i]) << std::endl;
  }
}

int main() {
    test_sparse_post_process();
  //  test_matmul();
  //  test_transpose();
  //  test_mask();
    test_sparse();


  return 0;
}








