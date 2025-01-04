#pragma once

#include "ckks_evaluator.cuh"
#include "row_pack.h"

namespace nexus {

inline PhantomPlaintext CKKSEncode(vector<double> data, shared_ptr<CKKSEvaluator> ckks_evaluator, PhantomCiphertext* ref_ct = nullptr) {
    PhantomPlaintext pt;
    if (ref_ct) {
        ckks_evaluator->encoder.encode(data, ref_ct->chain_index(), ref_ct->scale(), pt);
    } else {
        ckks_evaluator->encoder.encode(data, ckks_evaluator->scale, pt);
    }
    return pt;
}

inline PhantomCiphertext CKKSEncrypt(vector<double> data, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomCiphertext out;
    auto pt = CKKSEncode(data, ckks_evaluator);
    ckks_evaluator->encryptor.encrypt(pt, out);
    return out;
}

inline void assert_shape(Matrix x, int rows, int cols) {
    if (x.rows() != rows || x.cols() != cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }
}
inline void assert_shape(Vector x, int size) {
    if (x.size() != size) {
        throw std::invalid_argument("Vector dimension does not match");
    }
}

}  // namespace nexus
