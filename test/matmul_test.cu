#include "ciphertext.h"
#include "matrix_mul.cuh"
#include "phantom.h"

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>


using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;



size_t N = 1ULL << 16;
double SCALE = pow(2.0, 40);
size_t L = 8;

PhantomCiphertext enc(vector<double> data, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomPlaintext pt;
    ckks_evaluator->encoder.encode(data, SCALE, pt);
    PhantomCiphertext out;
    ckks_evaluator->encryptor.encrypt(pt, out);
    return out;
}

vector<double> dec(PhantomCiphertext ct, shared_ptr<CKKSEvaluator> ckks_evaluator) {
    PhantomPlaintext pt;
    ckks_evaluator->decryptor.decrypt(ct, pt);
    vector<double> out;
    ckks_evaluator->encoder.decode(pt, out);
    return out;
}

bool isClose(const vector<double>& v1, const vector<double>& v2, double rtol = 1e-3, double atol = 1e-3) {
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
    
    vector<int> TEST_COEFF_MODULI{60};
    for (int i=0; i<L; i++)
        TEST_COEFF_MODULI.push_back(40);
    TEST_COEFF_MODULI.push_back(60);

    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, TEST_COEFF_MODULI));

    auto context = make_shared<PhantomContext>(parms);
    auto secret_key = make_shared<PhantomSecretKey>(*context);
    auto public_key = make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
    auto relin_keys = make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = make_shared<PhantomGaloisKey>(secret_key->create_galois_keys(*context));

    auto encoder = make_shared<PhantomCKKSEncoder>(*context);

    auto ckks_evaluator = make_shared<CKKSEvaluator>(context, public_key, secret_key, encoder, relin_keys, galois_keys, SCALE);

    MMEvaluator mme(ckks_evaluator);

    /*
        ct1 = np.random.randn(128, 128)
        pt1 = np.random.randn(128, 128)
        ct2 = np.random.randn(128, 128)
        pt2 = np.random.randn(128, 128)
        
        ct_full = np.ones((SLOTS,))
        ct_full[:SLOTS//2] = ct1.flatten()
        ct_full[SLOTS//2:] = ct2.flatten()
        pt_full = np.ones((SLOTS,))
        pt_full[:SLOTS//2] = pt1.flatten()
        pt_full[SLOTS//2:] = pt2.flatten()

        res = multiply_ct128x128_pt128x128(ct_full, pt_full)
        assert np.isclose(res[:SLOTS//2], (ct1 @ pt1).flatten()).all()
        assert np.isclose(res[SLOTS//2:], (ct2 @ pt2).flatten()).all()
    */
    SECTION("ct 128x128 pt 128x128") {
        Matrix matrix_A1 = Matrix::Random(128, 128);
        Matrix matrix_B1 = Matrix::Random(128, 128);
        Matrix matrix_A2 = Matrix::Random(128, 128);
        Matrix matrix_B2 = Matrix::Random(128, 128);
        Matrix matrix_C1 = matrix_A1 * matrix_B1;
        Matrix matrix_C2 = matrix_A2 * matrix_B2;
        auto ct_matrix_128x128 = enc(mme.flatten_pack(matrix_A1, matrix_A2), ckks_evaluator);
        vector<double> pt_matrix_128x128 = mme.flatten_pack(matrix_B1, matrix_B2);
        vector<double> ctxpt_matrix_128x128 = mme.flatten_pack(matrix_C1, matrix_C2);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x128_pt128x128(ct_matrix_128x128, pt_matrix_128x128, res);
        auto mm_res = dec(res, ckks_evaluator);

        REQUIRE(isClose(mm_res, ctxpt_matrix_128x128));

        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x128_pt128x128(ct_matrix_128x128, pt_matrix_128x128, res);
        };
    }

    /*
        ct1 = np.random.randn(128, 128)
        ct2 = np.random.randn(128, 128)
        ct3 = np.random.randn(128, 128)
        ct1[:, 64:] = 0
        ct2[:, 64:] = 0
        ct3[:, 64:] = 0
        
        ct1_full = np.concat([ct1.flatten(), ct1.flatten()])
        ct2_full = np.concat([ct2.flatten(), ct2.flatten()])
        ct3_full = np.concat([ct3.flatten(), ct3.flatten()])

        gt_intern = ct1 @ ct2.transpose()
        gt = gt_intern @ ct3
        for i in range(gt_intern.shape[0]):
            gt_intern[i] = np.roll(gt_intern[i], -i)

        tmp = multiply_ct128x64_ct128x64_trans(ct1_full, ct2_full)
        for i, row in enumerate(tmp):
            assert np.isclose(tmp[:SLOTS//2], gt_intern.flatten()).all()
            assert np.isclose(tmp[SLOTS//2:], gt_intern.flatten()).all()
        res = multiply_ct128x128_ct128x128(tmp, ct3_full)
        assert np.isclose(res[:SLOTS//2], gt.flatten()).all()
    */
    SECTION("ct 128x64 ct.t 128x64 && ct 128x128 ct 128x64") {
        Matrix matrix1 = Matrix::Random(128, 128);
        Matrix matrix2 = Matrix::Random(128, 128);
        Matrix matrix3 = Matrix::Random(128, 128);
        Matrix placeholder = Matrix::Zero(128, 128);
        matrix1.block(0, 64, 128, 64).setZero();
        matrix2.block(0, 64, 128, 64).setZero();
        matrix3.block(0, 64, 128, 64).setZero();

        Matrix matrix_intermediate = matrix1 * matrix2.transpose();
        Matrix gt_matrix = matrix_intermediate * matrix3;
        for (int i = 0; i < matrix_intermediate.rows(); ++i) {
            Eigen::VectorXd row = matrix_intermediate.row(i);
            Eigen::VectorXd rotated_row(row.size());
            rotated_row << row.segment(i, row.size() - i), row.segment(0, i);
            matrix_intermediate.row(i) = rotated_row;
        }

        auto ct1 = enc(mme.flatten_pack(matrix1, matrix1), ckks_evaluator);
        auto ct2 = enc(mme.flatten_pack(matrix2, matrix2), ckks_evaluator);
        auto ct3 = enc(mme.flatten_pack(matrix3, matrix3), ckks_evaluator);
        vector<double> ct_matrix_intermediate = mme.flatten_pack(matrix_intermediate, matrix_intermediate);
        vector<double> ct_matrix_res = mme.flatten_pack(gt_matrix, placeholder);

        PhantomCiphertext res1, res2;
        mme.matrix_mul_ct128x64_ct128x64_transpose(ct1, ct2, res1);
        auto mm_res1 = dec(res1, ckks_evaluator);

        REQUIRE(isClose(mm_res1, ct_matrix_intermediate));

        mme.matrix_mul_ct128x128_ct128x128(res1, ct3, res2);
        auto mm_res2 = dec(res2, ckks_evaluator);

        REQUIRE(isClose(mm_res2, ct_matrix_res));

        BENCHMARK("matmul1") {
           mme.matrix_mul_ct128x64_ct128x64_transpose(ct1, ct2, res1);
        };
        
        BENCHMARK("matmul2") {
        mme.matrix_mul_ct128x128_ct128x128(res1, ct3, res2);
        };
    }

    /*
    np.random.seed(1104)
    ct = np.random.randn(128, 768)
    pt = np.random.randn(768, 128)
    
    cts = np.split(ct, 6, axis=1)
    cts = [
        (np.concat((cts[0].flatten(), cts[1].flatten()))), 
        (np.concat((cts[2].flatten(), cts[3].flatten()))), 
        (np.concat((cts[4].flatten(), cts[5].flatten()))), 
    ]
    pts = np.split(pt, 6, axis=0)
    pts = [
        (np.concat((pts[0].flatten(), pts[1].flatten()))), 
        (np.concat((pts[2].flatten(), pts[3].flatten()))), 
        (np.concat((pts[4].flatten(), pts[5].flatten()))), 
    ]

    res = multiply_ct128x768_pt768x128_type2(cts, pts)
    assert np.isclose(res[:SLOTS//2], (ct @ pt).flatten()).all()
    assert np.isclose(res[SLOTS//2:], (ct @ pt).flatten()).all()
    */
    SECTION("ct 128x768 pt 768x128") {
        Matrix matrix_A = Matrix::Random(128, 768);
        Matrix matrix_B = Matrix::Random(768, 128);
        Matrix matrix_C = matrix_A * matrix_B;
        vector<vector<double>> packed_A = mme.row_pack_128x768(matrix_A);
        vector<PhantomCiphertext> cts{
            enc(packed_A[0], ckks_evaluator), 
            enc(packed_A[1], ckks_evaluator), 
            enc(packed_A[2], ckks_evaluator), 
        };
        vector<vector<double>> pts = mme.column_pack_768x128(matrix_B);
        vector<double> ctxpt = mme.flatten_pack(matrix_C, matrix_C);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x768_pt768x128(cts, pts, res);
        auto mm_res = dec(res, ckks_evaluator);

        REQUIRE(isClose(mm_res, ctxpt));

        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x768_pt768x128(cts, pts, res);
        };
    }

    /*
    ct1 = np.random.randn(128, 768)
    ct2 = np.random.randn(768, 64)
    ct3 = np.random.randn(768, 64)
    
    ct1_full = row_pack_128x768(ct1)
    ct2_full = row_pack_768x64(ct2, ct3)

    gt = ct1 @ ct2

    res = multiply_ct128x768_pt768x128(ct1_full, ct2_full)
    assert np.isclose(res[:SLOTS//2].reshape((128, 128))[:, :64], ct1 @ ct2).all()
    assert np.isclose(res[SLOTS//2:].reshape((128, 128))[:, :64], ct1 @ ct3).all()
    */
    SECTION("ct 128x768 pt 768x64x2") {
        Matrix matrix_A = Matrix::Random(128, 768);
        Matrix matrix_B1 = Matrix::Random(768, 64);
        Matrix matrix_B2 = Matrix::Random(768, 64);
        Matrix matrix_C1 = matrix_A * matrix_B1;
        Matrix matrix_C2 = matrix_A * matrix_B2;
        vector<vector<double>> packed_A = mme.row_pack_128x768(matrix_A);
        vector<PhantomCiphertext> cts{
            enc(packed_A[0], ckks_evaluator), 
            enc(packed_A[1], ckks_evaluator), 
            enc(packed_A[2], ckks_evaluator), 
        };
        vector<vector<double>> pts = mme.row_pack_768x64x2(matrix_B1, matrix_B2);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x768_pt768x64x2(cts, pts, res);
        auto mm_res = dec(res, ckks_evaluator);

        Eigen::Map<Matrix> mm_res1(mm_res.data(), 128, 128);
        Eigen::Map<Matrix> mm_res2(mm_res.data() + 128*128, 128, 128);

        REQUIRE(matrix_C1.isApprox(mm_res1.block(0, 0, 128, 64), 1e-3));
        REQUIRE(matrix_C2.isApprox(mm_res2.block(0, 0, 128, 64), 1e-3));

        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x768_pt768x64x2(cts, pts, res);
        };
    }

    /*
    ct1 = np.random.randn(128, 768)
    ct2 = np.random.randn(768, 64)
    ct3 = np.random.randn(768, 64)
    
    ct1_full = row_pack_128x768(ct1)
    ct2_full = row_pack_768x64(ct2, ct3)

    gt = ct1 @ ct2

    res = multiply_ct128x768_pt768x128(ct1_full, ct2_full)
    assert np.isclose(res[:SLOTS//2].reshape((128, 128))[:, :64], ct1 @ ct2).all()
    assert np.isclose(res[SLOTS//2:].reshape((128, 128))[:, :64], ct1 @ ct3).all()
    */
    SECTION("ct 128x768 pt 768x768") {
        Matrix matrix_A = Matrix::Random(128, 768);
        Matrix matrix_B = Matrix::Random(768, 768);
        Matrix matrix_C = matrix_A * matrix_B;
        vector<vector<double>> packed_A = mme.row_pack_128x768(matrix_A);
        vector<PhantomCiphertext> cts{
            enc(packed_A[0], ckks_evaluator), 
            enc(packed_A[1], ckks_evaluator), 
            enc(packed_A[2], ckks_evaluator), 
        };
        auto pts = mme.row_pack_768x768(matrix_B);
        vector<PhantomCiphertext> res;
        mme.matrix_mul_ct128x768_pt768x768(cts, pts, res);

        Matrix mm_result = Matrix::Zero(128, 768);
        for (int i=0; i<3; i++) {
            auto mm_res = dec(res[i], ckks_evaluator);
            mm_result.block(0, i*256, 128, 128) = Eigen::Map<Matrix>(mm_res.data(), 128, 128);
            mm_result.block(0, i*256+128, 128, 128) = Eigen::Map<Matrix>(mm_res.data() + 128*128, 128, 128);
        }

        REQUIRE(mm_result.isApprox(matrix_C, 1e-3));

        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x768_pt768x768(cts, pts, res);
        };
    }

}