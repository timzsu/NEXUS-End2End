#pragma once

#include "nn/row_pack.h"
#include "nn/constant.cuh"
#include "ckks_evaluator.cuh"

namespace nexus {

typedef std::array<std::vector<double>, 256> PackedPt;
template <size_t d0>
using PackedPtArray = std::array<PackedPt, d0>;
template <size_t d0, size_t d1>
using PackedPtMat = std::array<PackedPtArray<d1>, d0>;

// constexpr size_t pt_length = N * (total_level + 2);

inline std::vector<double> rotate(std::vector<double>& x, int steps) {
  std::vector<double> out(x.size());
  for (int i = 0; i < x.size(); i++) {
    out[i] = x[(i+steps) % x.size()];
  }
  return out;
}

template<int size>
inline std::array<double, size> rotate(std::array<double, size>& x, int steps) {
  std::array<double, size> out;
  for (int i = 0; i < size; i++) {
    out[i] = x[(i+steps) % size];
  }
  return out;
}

PackedPt pt_pack(FlatVec& pt, shared_ptr<CKKSEvaluator> ckks);

template <size_t d0>
PackedPtArray<d0> pt_pack_1d(FlatVecArray pt, shared_ptr<CKKSEvaluator> ckks) {
  PackedPtArray<d0> packed_pts;
  TORCH_CHECK_EQ(pt.size(), d0);
  for (int i=0; i<d0; i++) {
    packed_pts[i] = pt_pack(pt[i], ckks);
  }
  return packed_pts;
}

template <size_t d0, size_t d1>
PackedPtMat<d0, d1> pt_pack_2d(FlatVecMat pt, shared_ptr<CKKSEvaluator> ckks) {
  PackedPtMat<d0, d1> packed_pts;
  TORCH_CHECK_EQ(pt.size(), d0);
  for (int i=0; i<d0; i++) {
    packed_pts[i] = pt_pack_1d<d1>(pt[i], ckks);
  }
  return packed_pts;
}

} // namespace nexus