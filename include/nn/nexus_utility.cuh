#pragma once

#include "ckks_evaluator.cuh"
#include <precompiled/torch_includes.h>

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


inline void assert_shape(torch::Tensor x, torch::IntArrayRef size) {
    TORCH_CHECK_EQ(x.sizes(), size);
}

inline void show(torch::Tensor x, torch::IntArrayRef boundary, std::string prefix) {
    std::vector<torch::indexing::TensorIndex> indices;
    for (auto &b:boundary) {
        indices.push_back(torch::indexing::Slice(0, b));
    }
    cerr << prefix << ": " << x.index(indices) << endl;
}

}  // namespace nexus
