#pragma once

#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>

namespace nexus {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::RowVectorXd Vector;

// basic pack
std::vector<double> flatten_pack(Matrix A);
std::vector<double> flatten_pack(Matrix A, Matrix B);

// matrix pack
std::vector<std::vector<double>> row_pack_128x768(Matrix matrix);
std::vector<std::vector<double>> row_pack_768x64x2(Matrix matrix1, Matrix matrix2);
std::vector<std::vector<std::vector<double>>> row_pack_768x768(Matrix matrix);
std::vector<std::vector<double>> row_pack_768x128(Matrix matrix);

// vector pack
std::vector<double> row_pack_128x1(Vector vector);
std::vector<double> row_pack_64x1x2(Vector vector1, Vector vector2);
std::vector<std::vector<double>> row_pack_768x1(Vector vector);

}