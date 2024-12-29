#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ckks_evaluator.cuh"
#include "matrix_mul.cuh"
#include "phantom.h"
#include "utils.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

// Choose test target here:
int TEST_TARGET_IDX = 4;

size_t N = 1ULL << 16;
size_t MM_LOG_N = 16;
size_t MM_N = 1ULL << MM_LOG_N;

double SCALE = pow(2.0, 40);

vector<string> TEST_TARGETS = {"MatMul", "Argmax", "SoftMax", "LayerNorm", "GELU"};
vector<vector<int>> COEFF_MODULI =
    {
        {60, 40, 60},                                                                      // MatMul (0)
        {17},                                                                              // Argmax (1) - Number of Moduli
        {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58},          // SoftMax (2)
        {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58},  // LayerNorm (3)
        {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58}   // GELU (4)
};

string TEST_TARGET = TEST_TARGETS[TEST_TARGET_IDX];
vector<int> TEST_COEFF_MODULI = COEFF_MODULI[TEST_TARGET_IDX];

void random_real(vector<double> &vec, size_t size) {
    random_device rn;
    mt19937_64 rnd(rn());
    thread_local std::uniform_real_distribution<double> distribution(-1, 1);

    for (size_t i = 0; i < size; i++) {
        vec[i] = distribution(rnd);
    }
}

void MM_test() {
    EncryptionParameters parms(scheme_type::ckks);

    parms.set_poly_modulus_degree(MM_N);
    parms.set_coeff_modulus(CoeffModulus::Create(MM_N, COEFF_MODULI[0]));

    auto context = std::make_shared<PhantomContext>(parms);
    auto secret_key = std::make_shared<PhantomSecretKey>(*context);
    auto public_key = std::make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
    auto relin_keys = std::make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = std::make_shared<PhantomGaloisKey>(secret_key->create_galois_keys(*context));

    auto encoder = std::make_shared<PhantomCKKSEncoder>(*context);

    auto ckks_evaluator = std::make_shared<CKKSEvaluator>(context, public_key, secret_key, encoder, relin_keys, galois_keys, SCALE);

    MMEvaluator mme(ckks_evaluator);

    // vector<double> ct_matrix_128x128(encoder->slot_count());
    // vector<double> pt_matrix_128x128(encoder->slot_count());
    // random_real(matrix_128x128, 128*128);
    
    auto ct_matrix_128x128 = mme.read_matrix(std::filesystem::path(__FILE__).parent_path() / "data/ct128x128.txt", 1, encoder->slot_count())[0];
    auto pt_matrix_128x128 = mme.read_matrix(std::filesystem::path(__FILE__).parent_path() / "data/pt128x128.txt", 1, encoder->slot_count())[0];

    // encrypt
    PhantomPlaintext pt;
    ckks_evaluator->encoder.encode(ct_matrix_128x128, SCALE, pt);
    PhantomCiphertext matrix_128x128_encrypted;
    ckks_evaluator->encryptor.encrypt(pt, matrix_128x128_encrypted);

    PhantomCiphertext res;
    auto timer = Timer();

    mme.matrix_mul_ct128x128_pt128x128(matrix_128x128_encrypted, pt_matrix_128x128, res);

    timer.stop();
    cout << "[MatMul] 128x128 x 128x128 takes: " << timer.duration<milliseconds>() << " milliseconds" << endl;

    auto ctxpt_res = mme.read_matrix(std::filesystem::path(__FILE__).parent_path() / "data/ct128x128_pt128x128.txt", 1, encoder->slot_count() / 2)[0];

    // Calculate the error of the first column
    PhantomPlaintext res_pt;
    vector<double> mm_res;
    ckks_evaluator->decryptor.decrypt(res, res_pt);
    ckks_evaluator->encoder.decode(res_pt, mm_res);

    double average_err = 0.0;
    for (auto i = 0; i < 128*128; i++) {
        average_err += fabs(mm_res[i] - ctxpt_res[i]);
    }
    cout << "Average Error: " << average_err / (128*128) << endl;
}

int main() {
    MM_test();
}