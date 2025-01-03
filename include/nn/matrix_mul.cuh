#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include "ckks_evaluator.cuh"
#include "phantom.h"
#include <Eigen/Core>

namespace nexus {
using namespace phantom;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

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
  */
  void matrix_mul_ct128x128_pt128x128(PhantomCiphertext& ct, vector<double>& pt, PhantomCiphertext &res);
  void matrix_mul_ct128x768_pt768x128(vector<PhantomCiphertext>& ct, vector<vector<double>>& pt, PhantomCiphertext &res) {
    PhantomCiphertext product, rotProduct;
    for (int i=0; i<3; i++) {
      matrix_mul_ct128x128_pt128x128(ct[i], pt[i], product);
      ckks->evaluator.rotate_vector(product, slot_count/2, *(ckks->galois_keys), rotProduct);
      ckks->evaluator.add_inplace(product, rotProduct);
      if (i == 0)
        res = product;
      else
        ckks->evaluator.add_inplace(res, product);
    }
  }
  void matrix_mul_ct128x768_pt768x64x2(vector<PhantomCiphertext>& ct, vector<vector<double>>& pt, PhantomCiphertext &res) {
    vector<PhantomCiphertext> buf0(3), buf1(3);
    for (int i=0; i<3; i++) {
      matrix_mul_ct128x128_pt128x128(ct[i],pt[i], buf0[i]);
      matrix_mul_ct128x128_pt128x128(ct[i],pt[i+3], buf1[i]);
    }
    PhantomCiphertext result0, result1;
    ckks->evaluator.add_many(buf0, result0);
    ckks->evaluator.add_many(buf1, result1);
    ckks->evaluator.rotate_vector_inplace(result1, slot_count / 2, *(ckks->galois_keys));
    ckks->evaluator.add(result0, result1, res);
  }
  void matrix_mul_ct128x768_pt768x768(vector<PhantomCiphertext>& ct, vector<vector<vector<double>>>& pt, vector<PhantomCiphertext> &res) {
    res.resize(3);
    for (int i=0; i<3; i++) {
      matrix_mul_ct128x768_pt768x64x2(ct, pt[i], res[i]);
    }
  }
  void matrix_mul_ct128x64_ct128x64_transpose(PhantomCiphertext& ct1, PhantomCiphertext& ct2, PhantomCiphertext &res);
  void matrix_mul_ct128x128_ct128x128(PhantomCiphertext& ct1, PhantomCiphertext& ct2, PhantomCiphertext &res);

  vector<double> flatten_pack(Matrix A, Matrix B) {
      vector<double> ct_matrix(A.data(), A.data() + A.size());
      vector<double> vec_B(B.data(), B.data() + B.size());
      ct_matrix.insert(ct_matrix.end(), vec_B.begin(), vec_B.end());
      return ct_matrix;
  }
  vector<vector<double>> row_pack_128x768(Matrix matrix) {
    return vector<vector<double>>{
      flatten_pack(matrix.block(0, 0, 128, 128), matrix.block(0, 128, 128, 128)), 
      flatten_pack(matrix.block(0, 2*128, 128, 128), matrix.block(0, 3*128, 128, 128)), 
      flatten_pack(matrix.block(0, 4*128, 128, 128), matrix.block(0, 5*128, 128, 128)), 
    };
  }
  vector<vector<double>> row_pack_768x64x2(Matrix matrix1, Matrix matrix2) {
    std::vector<Matrix> mat_res(6, Matrix::Zero(2*128, 128));

    for (int i = 0; i < 3; ++i) {
      mat_res[i].block(0, 0, 128, 64) = matrix1.block(i * 256, 0, 128, 64);
      mat_res[i].block(128, 0, 128, 64) = matrix2.block(i * 256+128, 0, 128, 64);
      mat_res[i+3].block(0, 0, 128, 64) = matrix2.block(i * 256, 0, 128, 64);
      mat_res[i+3].block(128, 0, 128, 64) = matrix1.block(i * 256+128, 0, 128, 64);
    }

    vector<vector<double>> result(6);
    for (int i = 0; i < 6; i++) {
      result[i] = vector<double>(mat_res[i].data(), mat_res[i].data() + mat_res[i].size());
    }
    return result;
  }
  vector<vector<vector<double>>> row_pack_768x768(Matrix matrix) {
    vector<vector<Matrix>> mat_res(3, vector<Matrix>(6, Matrix::Zero(2*128, 128)));
    vector<vector<vector<double>>> results(3);

    for (int m = 0; m < 3; ++m) {
      Matrix matrix1 = matrix.block(0, m * 256, 768, 128);
      Matrix matrix2 = matrix.block(0, m * 256 + 128, 768, 128);
      for (int i = 0; i < 3; ++i) {
        mat_res[m][i].block(0, 0, 128, 128) = matrix1.block(i * 256, 0, 128, 128);
        mat_res[m][i].block(128, 0, 128, 128) = matrix2.block(i * 256+128, 0, 128, 128);
        mat_res[m][i+3].block(0, 0, 128, 128) = matrix2.block(i * 256, 0, 128, 128);
        mat_res[m][i+3].block(128, 0, 128, 128) = matrix1.block(i * 256+128, 0, 128, 128);
      }
      results[m].resize(6);
      for (int i = 0; i < 6; i++) {
        results[m][i] = vector<double>(mat_res[m][i].data(), mat_res[m][i].data() + mat_res[m][i].size());
      }
    }

    return results;
  }
  vector<vector<double>> column_pack_768x128(Matrix matrix) {
    return vector<vector<double>>{
      flatten_pack(matrix.block(0, 0, 128, 128), matrix.block(128, 0, 128, 128)), 
      flatten_pack(matrix.block(2*128, 0, 128, 128), matrix.block(3*128, 0, 128, 128)), 
      flatten_pack(matrix.block(4*128, 0, 128, 128), matrix.block(5*128, 0, 128, 128)), 
    };
  }
};
}  // namespace nexus
