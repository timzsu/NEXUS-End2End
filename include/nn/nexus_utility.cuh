#pragma once

#include "Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include <cuComplex.h>
#include <precompiled/torch_includes.h>
#include "nn/constant.cuh"
#include "nn/row_pack.h"

namespace nexus {

// level: 1 ... L+1
inline uint64_t chain_idx(uint64_t level) {
    return total_level - level + 2;
}
inline uint64_t level_from_chain_idx(uint64_t chain_idx) {
    return total_level - chain_idx + 2;
}


template <class T>
inline PhantomPlaintext CKKSEncode(vector<T> data, shared_ptr<CKKSEvaluator> ckks_evaluator, PhantomCiphertext* ref_ct = nullptr, bool use_default_scale = false) {
    PhantomPlaintext pt;
    if (ref_ct) {
        auto scale = use_default_scale ? ckks_evaluator->scale : ref_ct->scale();
        ckks_evaluator->encoder.encode(data, scale, pt);
        ckks_evaluator->evaluator.mod_switch_to_inplace(pt, ref_ct->chain_index());
    } else {
        ckks_evaluator->encoder.encode(data, ckks_evaluator->scale, pt);
    }
    return pt;
}

template <class T>
inline PhantomCiphertext CKKSEncrypt(vector<T> data, shared_ptr<CKKSEvaluator> ckks_evaluator, int chain_index=boot_level+1) {
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
    auto mask_pt = CKKSEncode(mask, ckks, &res, true);
    ckks->evaluator.multiply_plain_inplace(res, mask_pt);
    ckks->evaluator.rescale_to_next_inplace(res);
    tmp = res;
    for (int i = 0; i < std::log2(len); ++i) {
        ckks->evaluator.rotate_vector(tmp, -pow(2, i), *ckks->galois_keys, res);
        ckks->evaluator.add_inplace(res, tmp);
        tmp = res;
    }
    return res;
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

inline void bootstrap(PhantomCiphertext &x, std::shared_ptr<Bootstrapper> bootstrapper, bool suppress_warning = false) {
    if (!suppress_warning && x.coeff_modulus_size() > 1) {
        printf("Warning: The ciphertext for bootstrap is at level %lu. Consider computation under lower levels to further improve performance. If you know this, pass suppress_warning=true to suppress this warning. \n", level_from_chain_idx(x.chain_index()));
    }
    while (x.coeff_modulus_size() > 1) {
        bootstrapper->ckks->evaluator.mod_switch_to_next_inplace(x);
    }
    PhantomCiphertext rtn;
    bootstrapper->set_final_scale(x.scale());
    bootstrapper->bootstrap_3(rtn, x);
    x = rtn;
}

inline void bootstrap(std::vector<PhantomCiphertext> &x, std::shared_ptr<Bootstrapper> bootstrapper) {
    for (auto& ct : x) {
        bootstrap(ct, bootstrapper);
    }
}

inline void mod_switch_to_same(PhantomCiphertext& x, PhantomCiphertext& y, std::shared_ptr<CKKSEvaluator> ckks_evaluator) {
    if (x.coeff_modulus_size() > y.coeff_modulus_size()) {
        ckks_evaluator->evaluator.mod_switch_to_inplace(x, y.chain_index());
    } else if (x.coeff_modulus_size() < y.coeff_modulus_size()) {
        ckks_evaluator->evaluator.mod_switch_to_inplace(y, x.chain_index());
    }
}

}  // namespace nexus
