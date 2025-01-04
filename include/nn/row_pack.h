#pragma once

#include <Eigen/Core>

namespace nexus {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::RowVectorXd Vector;
typedef std::vector<double> FlatVec;
typedef std::vector<FlatVec> FlatVecArray;
typedef std::vector<FlatVecArray> FlatVecMat;

// basic pack
FlatVec flatten_pack(Matrix A);
FlatVec flatten_pack(Matrix A, Matrix B);

// matrix pack
FlatVecArray row_pack_128x768(Matrix matrix);
FlatVecArray row_pack_768x64x2(Matrix matrix1, Matrix matrix2);
FlatVecMat row_pack_768x768(Matrix matrix);
FlatVecArray row_pack_768x128(Matrix matrix);

// vector pack
FlatVec row_pack_128x1(Vector vector);
FlatVec row_pack_64x1x2(Vector vector1, Vector vector2);
FlatVecArray row_pack_768x1(Vector vector);

}