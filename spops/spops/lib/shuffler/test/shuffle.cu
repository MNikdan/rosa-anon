#include "shuffle.h"

#include <algorithm>
#include <iostream>
#include <queue>

int main() {
  using T = int;
  int m = 6;
  int n = 6;
  const char *raw_mat =
//      "1000000000001000000000000000001001100000000000010101000000001000000010000000000000000100000000000000001000000000000000010000000000000000000000010000000000000100000000000000100000000000000001000000101000000000000010010000100000000001000000100000100000000000";

/*
  "1010000100000010"
  "0110001100001000"
  "1110000000100000"
  "0001000101000010"
  "0000101101010000"
  "0000011000001100"
  "0100111000010000"
  "1101100100000000"
  "0000000010010100"
  "0001100001000000"
  "0010000000101000"
  "0000101010010100"
  "0100010000101000"
  "0000010010010100"
  "1001000000000010"
  "0000000000000001"; */
      "100001"
      "000000"
      "001000"
      "000000"
      "010000"
      "100001";
//      "0000"
//      "0000"
//      "0000"
//      "0000"


  T *mat = new T[m * n];
  for (int i = 0; i < m * n; i++) {
    mat[i] = T(raw_mat[i] - '0');
  }

  Shuffle<T> shuffle{
      .rows = m,
      .cols = n,
      .mat = mat
  };

  shuffle.shuffle_cm();

  MatrixView view{.rows = m, .cols = n};

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << mat[view(shuffle.row_mapper[i], shuffle.col_mapper[j])] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  auto result = shuffle.remap(false);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << result[view(i, j)] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}



