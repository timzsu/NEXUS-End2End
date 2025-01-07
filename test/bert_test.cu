#include "bert/bert.cuh"

#include <precompiled/catch2_includes.h>
#include <precompiled/torch_includes.h>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

constexpr double MAX_RTOL=1e-3;
constexpr double MAX_ATOL=1e-2;

PhantomCiphertext enc(vector<double> data, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomPlaintext pt;
    PhantomCiphertext ct;
    ckks_evaluator->encoder.encode(data, ckks_evaluator->scale, pt);
    ckks_evaluator->encryptor.encrypt(pt, ct);
    return ct;
}

vector<double> dec(PhantomCiphertext ct, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomPlaintext pt;
    ckks_evaluator->decryptor.decrypt(ct, pt);
    vector<double> out;
    ckks_evaluator->encoder.decode(pt, out);
    return out;
}

torch::Tensor random_tensor(torch::IntArrayRef size, double min, double max) {
    return torch::rand(size, torch::kDouble) * (max - min) + min;   
}

TEST_CASE("BERT Components") {
    long logN = 16;
    long logn = logN - 2;
    long sparse_slots = (1 << logn);

    int logp = 50;
    int logq = 51;
    int log_special_prime = 51;

    // QuickMax: 17
    int main_mod_count = 17;

    // Bootstrapping costs 14 mods: subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2
    int bs_mod_count = 14;

    int total_level = main_mod_count + bs_mod_count;
    int secret_key_hamming_weight = 192;

    // Bootstrapping parameters
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;
    long loge = 10;

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < main_mod_count; i++) {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < bs_mod_count; i++) {
        coeff_bit_vec.push_back(logq);
    }
    coeff_bit_vec.push_back(log_special_prime);

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    auto context = std::make_shared<PhantomContext>(parms);
    auto secret_key = std::make_shared<PhantomSecretKey>(*context);
    auto public_key = std::make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
    auto relin_keys = std::make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = std::make_shared<PhantomGaloisKey>();

    auto encoder = std::make_shared<PhantomCKKSEncoder>(*context);

    auto ckks_evaluator = std::make_shared<CKKSEvaluator>(context, public_key, secret_key, encoder, relin_keys, galois_keys, scale);

    SECTION("Attention") {
        BertAttention attention(ckks_evaluator);

        torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
        auto packed_input = row_pack_128x768(input);
        std::vector<PhantomCiphertext> input_ct;
        for (auto &inp : packed_input) {
            input_ct.push_back(enc(inp, ckks_evaluator));
        }

        auto out = attention.forward(input_ct);

        for (auto &o : out) {
            auto dec_out = dec(o, ckks_evaluator);
            auto tensor_out = tensor_from_vector(dec_out, 256, 128);
        }
    }
    
    SECTION("MLP") {
        BertMLP mlp(ckks_evaluator);

        torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
        auto packed_input = row_pack_128x768(input);
        std::vector<PhantomCiphertext> input_ct;
        for (auto &inp : packed_input) {
            input_ct.push_back(enc(inp, ckks_evaluator));
        }

        auto out = mlp.forward(input_ct);

        for (auto &o : out) {
            auto dec_out = dec(o, ckks_evaluator);
            auto tensor_out = tensor_from_vector(dec_out, 256, 128);
        }
    }
}