#include "matrix_mul.cuh"
#include "phantom.h"

#include <Eigen/Core>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>


using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

size_t N = 1ULL << 16;
double SCALE = pow(2.0, 40);
vector<int> TEST_COEFF_MODULI{60, 40, 40, 40, 40, 60};

vector<double> flatten_pack(Matrix& A, Matrix& B) {
    std::vector<double> ct_matrix(A.data(), A.data() + A.size());
    std::vector<double> vec_B(B.data(), B.data() + B.size());
    ct_matrix.insert(ct_matrix.end(), vec_B.begin(), vec_B.end());
    return ct_matrix;
}

PhantomCiphertext enc(std::vector<double> data, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomPlaintext pt;
    ckks_evaluator->encoder.encode(data, SCALE, pt);
    PhantomCiphertext out;
    ckks_evaluator->encryptor.encrypt(pt, out);
    return out;
}

std::vector<double> dec(PhantomCiphertext ct, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomPlaintext pt;
    ckks_evaluator->decryptor.decrypt(ct, pt);
    std::vector<double> out;
    ckks_evaluator->encoder.decode(pt, out);
    return out;
}

bool isClose(const std::vector<double>& v1, const std::vector<double>& v2, double rtol = 1e-3, double atol = 1e-3) {
    if (v1.size() != v2.size()) {
        return false;
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = fabs(v1[i] - v2[i]);
        double tol = atol + rtol * fabs(v2[i]);
        if (diff > tol) {
            cerr << "diff=" << diff << " tol=" << tol << endl;
            return false;
        }
    }
    return true;
}

TEST_CASE("Matrix Multiplication") {
    
    EncryptionParameters parms(scheme_type::ckks);

    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, TEST_COEFF_MODULI));

    auto context = std::make_shared<PhantomContext>(parms);
    auto secret_key = std::make_shared<PhantomSecretKey>(*context);
    auto public_key = std::make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
    auto relin_keys = std::make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = std::make_shared<PhantomGaloisKey>(secret_key->create_galois_keys(*context));

    auto encoder = std::make_shared<PhantomCKKSEncoder>(*context);

    auto ckks_evaluator = std::make_shared<CKKSEvaluator>(context, public_key, secret_key, encoder, relin_keys, galois_keys, SCALE);

    MMEvaluator mme(ckks_evaluator);

    SECTION("ct 128x128 pt 128x128") {
        Matrix matrix_A1 = Matrix::Random(128, 128);
        Matrix matrix_B1 = Matrix::Random(128, 128);
        Matrix matrix_A2 = Matrix::Random(128, 128);
        Matrix matrix_B2 = Matrix::Random(128, 128);
        Matrix matrix_C1 = matrix_A1 * matrix_B1;
        Matrix matrix_C2 = matrix_A2 * matrix_B2;
        auto ct_matrix_128x128 = enc(flatten_pack(matrix_A1, matrix_A2), ckks_evaluator);
        std::vector<double> pt_matrix_128x128 = flatten_pack(matrix_B1, matrix_B2);
        std::vector<double> ctxpt_matrix_128x128 = flatten_pack(matrix_C1, matrix_C2);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x128_pt128x128(ct_matrix_128x128, pt_matrix_128x128, res);
        auto mm_res = dec(res, ckks_evaluator);

        REQUIRE(isClose(mm_res, ctxpt_matrix_128x128));

        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x128_pt128x128(ct_matrix_128x128, pt_matrix_128x128, res);
        };
    }

    SECTION("ct 128x64 ct.t 128x64 && ct 128x128 ct 128x64") {
        Matrix matrix1 = Matrix::Random(128, 128);
        Matrix matrix2 = Matrix::Random(128, 128);
        matrix1.block(0, 64, 128, 64).setZero();
        matrix2.block(0, 64, 128, 64).setZero();

        Matrix matrix_intermediate = matrix1 * matrix2.transpose();
        for (int i = 0; i < matrix_intermediate.rows(); ++i) {
            Eigen::VectorXd row = matrix_intermediate.row(i);
            Eigen::VectorXd rotated_row(row.size());
            rotated_row << row.segment(i, row.size() - i), row.segment(0, i);
            matrix_intermediate.row(i) = rotated_row;
        }

        auto ct1 = enc(flatten_pack(matrix1, matrix1), ckks_evaluator);
        auto ct2 = enc(flatten_pack(matrix2, matrix2), ckks_evaluator);
        std::vector<double> ct_matrix_intermediate = flatten_pack(matrix_intermediate, matrix_intermediate);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x64_ct128x64_transpose(ct1, ct2, res);
        auto mm_res = dec(res, ckks_evaluator);

        REQUIRE(isClose(mm_res, ct_matrix_intermediate));

        BENCHMARK("matmul") {
           mme.matrix_mul_ct128x64_ct128x64_transpose(ct1, ct2, res);
        };
    }

}