#include "nn/row_pack.h"
#include <Eigen/Core>

namespace nexus {

using namespace std;

FlatVec flatten_pack(Matrix A) {
    FlatVec ct_matrix(A.data(), A.data() + A.size());
    auto buf = ct_matrix;
    ct_matrix.insert(ct_matrix.end(), buf.begin(), buf.end());
    return ct_matrix;
}

FlatVec flatten_pack(Matrix A, Matrix B) {
    FlatVec ct_matrix(A.data(), A.data() + A.size());
    FlatVec vec_B(B.data(), B.data() + B.size());
    ct_matrix.insert(ct_matrix.end(), vec_B.begin(), vec_B.end());
    return ct_matrix;
}

FlatVecArray row_pack_128x768(Matrix matrix) {
    return FlatVecArray{
        flatten_pack(matrix.block(0, 0, 128, 128), matrix.block(0, 128, 128, 128)), 
        flatten_pack(matrix.block(0, 2*128, 128, 128), matrix.block(0, 3*128, 128, 128)), 
        flatten_pack(matrix.block(0, 4*128, 128, 128), matrix.block(0, 5*128, 128, 128)), 
    };
}

FlatVecArray row_pack_768x64x2(Matrix matrix1, Matrix matrix2) {
    vector<Matrix> mat_res(6, Matrix::Zero(2*128, 128));

    for (int i = 0; i < 3; ++i) {
      mat_res[i].block(0, 0, 128, 64) = matrix1.block(i * 256, 0, 128, 64);
      mat_res[i].block(128, 0, 128, 64) = matrix2.block(i * 256+128, 0, 128, 64);
      mat_res[i+3].block(0, 0, 128, 64) = matrix2.block(i * 256, 0, 128, 64);
      mat_res[i+3].block(128, 0, 128, 64) = matrix1.block(i * 256+128, 0, 128, 64);
    }

    FlatVecArray result(6);
    for (int i = 0; i < 6; i++) {
      result[i] = FlatVec(mat_res[i].data(), mat_res[i].data() + mat_res[i].size());
    }
    return result;
  }

FlatVecMat row_pack_768x768(Matrix matrix) {
    vector<vector<Matrix>> mat_res(3, vector<Matrix>(6, Matrix::Zero(2*128, 128)));
    FlatVecMat results(3);

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
        results[m][i] = FlatVec(mat_res[m][i].data(), mat_res[m][i].data() + mat_res[m][i].size());
        }
    }

    return results;
}

FlatVecArray row_pack_768x128(Matrix matrix) {
    return FlatVecArray{
        flatten_pack(matrix.block(0, 0, 128, 128), matrix.block(128, 0, 128, 128)), 
        flatten_pack(matrix.block(2*128, 0, 128, 128), matrix.block(3*128, 0, 128, 128)), 
        flatten_pack(matrix.block(4*128, 0, 128, 128), matrix.block(5*128, 0, 128, 128)), 
    };
}

FlatVec row_pack_128x1(Vector vector) {
    Matrix buf = Matrix::Zero(2*128, 128);
    buf.rowwise() = vector;
    return FlatVec(buf.data(), buf.data() + buf.size());
}
FlatVec row_pack_64x1x2(Vector vector1, Vector vector2) {
    Matrix buf = Matrix::Zero(2*128, 128);
    buf.block(0, 0, 128, 64).rowwise() = vector1;
    buf.block(128, 0, 128, 64).rowwise() = vector2;
    return FlatVec(buf.data(), buf.data() + buf.size());
}
FlatVecArray row_pack_768x1(Vector vector) {
    std::vector<Matrix> mm_res(3, Matrix::Zero(2*128, 128));
    for (int m=0; m<3; m++) {
        Vector vector1 = vector.segment(m * 256, 128);
        Vector vector2 = vector.segment(m * 256 + 128, 128);
        mm_res[m].block(0, 0, 128, 128).rowwise() = vector1;
        mm_res[m].block(128, 0, 128, 128).rowwise() = vector2;
    }
    FlatVecArray result(3);
    for (int i = 0; i < 3; i++) {
        result[i] = FlatVec(mm_res[i].data(), mm_res[i].data() + mm_res[i].size());
    }
    return result;
}

} // namespace nexus