#pragma once

#include "ckks_evaluator.cuh"
#include <precompiled/torch_includes.h>
#include "nn/constant.cuh"
#include "nn/row_pack.h"

namespace nexus {

inline PhantomCiphertext quick_sum(const PhantomCiphertext& x, std::shared_ptr<CKKSEvaluator> ckks, int len) {
    PhantomCiphertext tmp = x, res;
    std::vector<double> mask(slot_count, 0);
    for (int i=0; i<slot_count; i+=128) {
        mask[i] = 1;
    }
    for (int i = 0; i < std::log2(len); ++i) {
        ckks->evaluator.rotate_vector(tmp, pow(2, i), *ckks->galois_keys, res);
        ckks->evaluator.add_inplace(res, tmp);
        tmp = res;
    }
    ckks->evaluator.multiply_vector_inplace_reduced_error(res, mask);
    ckks->evaluator.rescale_to_next_inplace(res);
    tmp = res;
    for (int i = 0; i < std::log2(len); ++i) {
        ckks->evaluator.rotate_vector(tmp, -pow(2, i), *ckks->galois_keys, res);
        ckks->evaluator.add_inplace(res, tmp);
        tmp = res;
    }
    return res;
}

inline PhantomPlaintext CKKSEncode(vector<double> data, shared_ptr<CKKSEvaluator> ckks_evaluator, PhantomCiphertext* ref_ct = nullptr) {
    PhantomPlaintext pt;
    if (ref_ct) {
        ckks_evaluator->encoder.encode(data, ref_ct->chain_index(), ref_ct->scale(), pt);
    } else {
        ckks_evaluator->encoder.encode(data, ckks_evaluator->scale, pt);
    }
    return pt;
}

inline PhantomCiphertext CKKSEncrypt(vector<double> data, shared_ptr<CKKSEvaluator> ckks_evaluator, int chain_index=boot_level+1) {
    PhantomCiphertext out;
    auto pt = CKKSEncode(data, ckks_evaluator);
    ckks_evaluator->encryptor.encrypt(pt, out);
    if (chain_index > 1) {
        ckks_evaluator->evaluator.mod_switch_to_inplace(out, chain_index);
    }
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

inline torch::Tensor tensor_from_ciphertexts(std::vector<PhantomCiphertext>& ciphertexts, std::shared_ptr<CKKSEvaluator> ckks_evaluator) {
    std::vector<torch::Tensor> decrypted_out;
    for (auto &o : ciphertexts) {
        auto tensor_out = tensor_from_vector(CKKSDecrypt(o, ckks_evaluator), {2, 128, 128});
        decrypted_out.push_back(tensor_out.index({0}));
        decrypted_out.push_back(tensor_out.index({1}));
    }
    return torch::concat(decrypted_out, -1);
}

}  // namespace nexus
