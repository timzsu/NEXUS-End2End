#pragma once

#include "ATen/core/ATen_fwd.h"
#include <precompiled/torch_includes.h>

namespace nexus {

typedef std::vector<double> FlatVec;
typedef std::vector<FlatVec> FlatVecArray;
typedef std::vector<FlatVecArray> FlatVecMat;

// Convert a torch::Tensor to a FlatVec (std::vector<double>)
inline FlatVec vector_from_tensor(torch::Tensor t) {
    TORCH_CHECK(t.dtype() == torch::kDouble);
    t = t.to(torch::kCPU).contiguous();
    return FlatVec(t.const_data_ptr<double>(), t.const_data_ptr<double>() + t.numel());
}

inline torch::Tensor tensor_from_vector(FlatVec vec, torch::IntArrayRef size) {
    return torch::from_blob(vec.data(), size, torch::kDouble).clone();
}

// basic pack

// Flatten and concatenate a single tensor
inline FlatVec flatten_pack(torch::Tensor A) {
    FlatVec ct_matrix = vector_from_tensor(A);
    auto buf = ct_matrix;
    ct_matrix.insert(ct_matrix.end(), buf.begin(), buf.end());
    return ct_matrix;
}
// Flatten and concatenate two tensors
inline FlatVec flatten_pack(torch::Tensor A, torch::Tensor B) {
    FlatVec ct_matrix = vector_from_tensor(A);
    FlatVec vec_B = vector_from_tensor(B);
    ct_matrix.insert(ct_matrix.end(), vec_B.begin(), vec_B.end());
    return ct_matrix;
}

// matrix pack
FlatVecArray row_pack_128x768(torch::Tensor matrix);
FlatVecArray row_pack_768x64x2(torch::Tensor matrix1, torch::Tensor matrix2);
FlatVecMat row_pack_768x768(torch::Tensor matrix);
FlatVecArray row_pack_768x128(torch::Tensor matrix);

// vector pack
FlatVec row_pack_128x1(torch::Tensor vector);
FlatVec row_pack_64x1x2(torch::Tensor vector1, torch::Tensor vector2);
FlatVecArray row_pack_768x1(torch::Tensor vector);

}