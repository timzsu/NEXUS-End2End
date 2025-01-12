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
    auto poly_modulus_degree = 1ULL << 16;
    double scale = pow(2.0, 40);
    EncryptionParameters parms(scheme_type::ckks);
    
    vector<int> coeff_modulus{60};
    for (int i=0; i<22; i++)
        coeff_modulus.push_back(40);
    coeff_modulus.push_back(60);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_modulus));

    auto context = std::make_shared<PhantomContext>(parms);
    auto secret_key = std::make_shared<PhantomSecretKey>(*context);
    auto public_key = std::make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
    auto relin_keys = std::make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = std::make_shared<PhantomGaloisKey>(secret_key->create_galois_keys(*context));

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

        attention.pack_weights();

        auto out = attention.forward(input_ct);

        for (auto &o : out) {
            auto dec_out = dec(o, ckks_evaluator);
            auto tensor_out = tensor_from_vector(dec_out, {256, 128});
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

        mlp.pack_weights();

        auto out = mlp.forward(input_ct);

        for (auto &o : out) {
            auto dec_out = dec(o, ckks_evaluator);
            auto tensor_out = tensor_from_vector(dec_out, {256, 128});
        }
    }
}