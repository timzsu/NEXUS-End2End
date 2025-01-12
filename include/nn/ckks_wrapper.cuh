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

inline vector<double> CKKSDecrypt(PhantomCiphertext ct, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomPlaintext pt;
    ckks_evaluator->decryptor.decrypt(ct, pt);
    vector<double> out;
    ckks_evaluator->encoder.decode(pt, out);
    return out;
}


inline void assert_shape(torch::Tensor x, int rows, int cols) {
    TORCH_CHECK_EQ(x.size(0), rows);
    TORCH_CHECK_EQ(x.size(1), cols);
}
inline void assert_shape(torch::Tensor x, int size) {
    TORCH_CHECK_EQ(x.size(0), size);
}

}  // namespace nexus
