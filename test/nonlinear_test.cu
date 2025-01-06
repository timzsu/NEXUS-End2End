#include "nn/ckks_wrapper.cuh"
#include "nn/row_pack.h"

#include "nn/softmax.cuh"
#include "nn/gelu.cuh"
#include "nn/layer_norm.cuh"
#include "nn/argmax.cuh"

#include <precompiled/catch2_includes.h>
#include <precompiled/torch_includes.h>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;


size_t N = 1ULL << 16;
double SCALE = pow(2.0, 40);
size_t L = 18;
constexpr double MAX_RTOL=1e-3;
constexpr double MAX_ATOL=1e-2;

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

TEST_CASE("Non-linear Operations") {
    
    EncryptionParameters parms(scheme_type::ckks);
    
    vector<int> coeff_modulus{60};
    for (int i=0; i<L; i++)
        coeff_modulus.push_back(40);
    coeff_modulus.push_back(60);

    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_modulus));

    auto context = make_shared<PhantomContext>(parms);
    auto secret_key = make_shared<PhantomSecretKey>(*context);
    auto public_key = make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
    auto relin_keys = make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = make_shared<PhantomGaloisKey>(secret_key->create_galois_keys(*context));

    auto encoder = make_shared<PhantomCKKSEncoder>(*context);

    auto ckks_evaluator = make_shared<CKKSEvaluator>(context, public_key, secret_key, encoder, relin_keys, galois_keys, SCALE);

    SECTION("Softmax") {
        SoftmaxEvaluator softmax_evaluator(ckks_evaluator);

        torch::Tensor matrix_A = random_tensor({128, 128}, -1, 1);
        matrix_A -= std::get<0>(matrix_A.max(1, true));
        torch::Tensor matrix_res = torch::softmax(matrix_A, 1);

        PhantomCiphertext ct_matrix = CKKSEncrypt(vector_from_tensor(torch::concat({matrix_A, torch::zeros_like(matrix_A)}, 1)), ckks_evaluator);
        PhantomCiphertext res;
        softmax_evaluator.softmax(ct_matrix, res, 128);
        auto mm_res = dec(res, ckks_evaluator);
        torch::Tensor tensor_res = tensor_from_vector(mm_res, 128, 256);

        REQUIRE(torch::allclose(tensor_res.slice(1, 0, 128), matrix_res, MAX_RTOL, MAX_ATOL));

        BENCHMARK("softmax") {
            softmax_evaluator.softmax(ct_matrix, res, 128);
        };
    }

    SECTION("GELU") {
        GELUEvaluator gelu_evaluator(ckks_evaluator);

        torch::Tensor matrix_A = torch::randn({128, 256}, torch::kDouble);
        torch::Tensor matrix_res = torch::nn::functional::gelu(matrix_A);
        PhantomCiphertext ct_matrix = CKKSEncrypt(vector_from_tensor(matrix_A), ckks_evaluator);

        PhantomCiphertext res;
        gelu_evaluator.gelu(ct_matrix, res);
        auto mm_res = dec(res, ckks_evaluator);
        torch::Tensor tensor_res = tensor_from_vector(mm_res, 128, 256);

        REQUIRE(torch::allclose(matrix_res, tensor_res, MAX_RTOL, MAX_ATOL));

        BENCHMARK("gelu") {
            gelu_evaluator.gelu(ct_matrix, res);
        };
    }

    SECTION("Layer Norm") {
        LNEvaluator ln_evaluator(ckks_evaluator);

        torch::Tensor matrix_A = random_tensor({16, 768}, -3, 3);
        matrix_A -= matrix_A.mean(1, true);
        torch::Tensor placeholder = torch::zeros({16, 2048-768}, torch::kDouble);
        torch::Tensor matrix_res = torch::layer_norm(matrix_A, 768);
        PhantomCiphertext ct_matrix = CKKSEncrypt(vector_from_tensor(torch::concat({matrix_A, placeholder}, 1)), ckks_evaluator);

        PhantomCiphertext res;
        ln_evaluator.layer_norm(ct_matrix, res, 1024);
        auto mm_res = dec(res, ckks_evaluator);
        torch::Tensor tensor_res = tensor_from_vector(mm_res, 16, 2048);

        REQUIRE(torch::allclose(tensor_res.slice(1, 0, 768), matrix_res, MAX_RTOL, MAX_ATOL));

        BENCHMARK("layer_norm") {
            ln_evaluator.layer_norm(ct_matrix, res, 1024);
        };
    }
}

TEST_CASE("Argmax") {
    long logN = 15;
    long logn = logN - 2;
    long sparse_slots = (1 << logn);

    int logp = 46;
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
    auto bootstrapper = std::make_shared<Bootstrapper>(
        loge,
        logn,
        logN - 1,
        total_level,
        scale,
        boundary_K,
        deg,
        scale_factor,
        inverse_deg,
        ckks_evaluator);
    ArgmaxEvaluator argmax_evaluator(ckks_evaluator, bootstrapper, main_mod_count);

    // Read Argmax input
    size_t slot_count = encoder->slot_count();

    PhantomPlaintext plain_input;
    PhantomCiphertext cipher_input;
    PhantomCiphertext cipher_output;
    vector<double> input(slot_count, 0.0);

    int argmax_input_size = 8;
    torch::Tensor input_tensor = torch::zeros({sparse_slots}, torch::kDouble);
    input_tensor.slice(0, 0, argmax_input_size) = random_tensor({argmax_input_size}, -0.5, 0.5);
    torch::Tensor output_tensor = torch::zeros({sparse_slots}, torch::kDouble);
    output_tensor.index({torch::argmax(input_tensor)}) = 1.0;

    vector<double> argmax_input = vector_from_tensor(input_tensor);

    // Sparse input
    for (size_t i = 0; i < slot_count; i++) {
        input[i] = argmax_input[i % sparse_slots];
    }

    // Initialize the bootstrapper
    cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper->prepare_mod_polynomial();

    cout << "Adding Bootstrapping Keys..." << endl;
    vector<int> gal_steps_vector;

    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    bootstrapper->addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    gal_steps_vector.push_back(-argmax_input_size);
    int log_step = log2(argmax_input_size);
    for (int i = 0; i < log_step; ++i) {
        gal_steps_vector.push_back(pow(2, i));
    }
    ckks_evaluator->decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator->galois_keys));
    bootstrapper->slot_vec.push_back(logn);

    cout << "Generating Linear Transformation Coefficients..." << endl;
    bootstrapper->generate_LT_coefficient_3();

    ckks_evaluator->encoder.encode(input, scale, plain_input);
    ckks_evaluator->encryptor.encrypt(plain_input, cipher_input);
    ckks_evaluator->encryptor.encrypt(plain_input, cipher_output); // TODO: change
    for (int i = 0; i < bs_mod_count; i++) {
        ckks_evaluator->evaluator.mod_switch_to_next_inplace(cipher_input);
    }

    argmax_evaluator.argmax(cipher_input, cipher_output, argmax_input_size);
    auto mm_res = dec(cipher_output, ckks_evaluator);
    torch::Tensor tensor_res = tensor_from_vector(mm_res, 2, sparse_slots);

    REQUIRE(torch::allclose(tensor_res.index({0}).slice(0, 0, argmax_input_size), output_tensor.slice(0, 0, argmax_input_size), 5e-2, 5e-2));

}