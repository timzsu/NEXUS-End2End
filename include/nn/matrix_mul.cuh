#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include "ckks_evaluator.cuh"

namespace nexus {
using namespace phantom;

class MMEvaluator {
 private:
  std::shared_ptr<CKKSEvaluator> ckks;

  void enc_compress_ciphertext(vector<double> &values, PhantomCiphertext &ct);
  vector<PhantomCiphertext> decompress_ciphertext(PhantomCiphertext &encrypted);

 public:
  MMEvaluator(std::shared_ptr<CKKSEvaluator> ckks) : ckks(ckks) {}

  // Helper functions
  inline vector<vector<double>> read_matrix(const std::string &filename, int rows, int cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::ifstream file(filename);

    if (!file.is_open()) {
      std::cerr << "Can not open file: " << filename << std::endl;
      return matrix;
    }

    std::string line;
    for (int i = 0; i < rows; ++i) {
      if (std::getline(file, line)) {
        std::istringstream iss(line);
        for (int j = 0; j < cols; ++j) {
          if (!(iss >> matrix[i][j])) {
            std::cerr << "read error: " << filename << " (row: " << i << ", column: " << j << ")" << std::endl;
          }
        }
      }
    }

    file.close();
    return matrix;
  }

  inline vector<vector<double>> transpose_matrix(const vector<vector<double>> &matrix) {
    if (matrix.empty()) {
      return {};
    }
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<double>> transposedMatrix(cols, std::vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        transposedMatrix[j][i] = matrix[i][j];
      }
    }

    return transposedMatrix;
  }

  // Evaluation function
  void matrix_mul(vector<vector<double>> &x, vector<vector<double>> &y, vector<PhantomCiphertext> &res);
  void multiply_power_of_x(PhantomCiphertext &encrypted, PhantomCiphertext &destination, int index);

  // NEXUS-specific function
  static constexpr size_t slot_count = 32768;
  /*
  Two simutaneous ct-pt multiplications with 128x128 matrices. 
  @Syntax: ct = ct1 | ct2, pt = pt1 | pt2 -> ct1 * pt1 | ct2 * pt2
  @note: ct1, ct2, pt1, pt2 ∈ ℝ^{128x128}
  */
  void matrix_mul_ct128x128_pt128x128(PhantomCiphertext& ct, vector<double>& pt, PhantomCiphertext &res);
  void matrix_mul_ct128x768_pt768x128(vector<PhantomCiphertext>& ct, vector<vector<double>>& pt, PhantomCiphertext &res);
  void matrix_mul_ct128x768_pt768x64x2(vector<PhantomCiphertext>& ct, vector<vector<double>>& pt, PhantomCiphertext &res);
  void matrix_mul_ct128x768_pt768x768(vector<PhantomCiphertext>& ct, vector<vector<vector<double>>>& pt, vector<PhantomCiphertext> &res);
  void matrix_mul_ct128x64_ct128x64_transpose(PhantomCiphertext& ct1, PhantomCiphertext& ct2, PhantomCiphertext &res);
  void matrix_mul_ct128x128_ct128x128(PhantomCiphertext& ct1, PhantomCiphertext& ct2, PhantomCiphertext &res);
};

}  // namespace nexus
