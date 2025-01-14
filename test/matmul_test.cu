#include "nn/matrix_mul.cuh"
#include "nn/nexus_utility.cuh"
#include "nn/row_pack.h"
#include "torch/cuda.h"

#include <precompiled/catch2_includes.h>
#include <precompiled/torch_includes.h>


using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;


size_t N = 1ULL << 16;
double SCALE = pow(2.0, 40);
size_t L = 8;
constexpr double MAX_RTOL=1e-3;
constexpr double MAX_ATOL=1e-3;

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
        torch::Tensor matrix_A1 = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix_B1 = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix_A2 = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix_B2 = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix_C1 = torch::mm(matrix_A1, matrix_B1);
        torch::Tensor matrix_C2 = torch::mm(matrix_A2, matrix_B2);
        auto ct_matrix = CKKSEncrypt(flatten_pack(matrix_A1, matrix_A2), ckks_evaluator);
        vector<double> pt_matrix = flatten_pack(matrix_B1, matrix_B2);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x128_pt128x128(ct_matrix, pt_matrix, res);
        CHECK(res.chain_index() == ct_matrix.chain_index() + 1);
        auto mm_res = tensor_from_vector(CKKSDecrypt(res, ckks_evaluator), {2, 128, 128});

        CHECK(torch::allclose(mm_res.index({0}), matrix_C1, MAX_RTOL, MAX_ATOL));
        CHECK(torch::allclose(mm_res.index({1}), matrix_C2, MAX_RTOL, MAX_ATOL));

        torch::cuda::synchronize();
        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x128_pt128x128(ct_matrix, pt_matrix, res);
            torch::cuda::synchronize();
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
        torch::Tensor matrix1A = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix1B = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix2A = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix2B = torch::randn({128, 128}, torch::kDouble);
        torch::Tensor matrix3A = torch::randn({128, 64}, torch::kDouble);
        torch::Tensor matrix3B = torch::randn({128, 64}, torch::kDouble);
        torch::Tensor matrix3 = torch::cat({matrix3A, matrix3B}, 1);
        matrix1A.slice(1, 64, 128) = 0;
        matrix1B.slice(1, 64, 128) = 0;
        matrix2A.slice(1, 64, 128) = 0;
        matrix2B.slice(1, 64, 128) = 0;

        torch::Tensor matrix_intermediate_A = torch::mm(matrix1A, matrix2A.transpose(0, 1));
        torch::Tensor matrix_intermediate_B = torch::mm(matrix1B, matrix2B.transpose(0, 1));
        torch::Tensor gt_matrix_A = torch::mm(matrix_intermediate_A, matrix3A);
        torch::Tensor gt_matrix_B = torch::mm(matrix_intermediate_B, matrix3B);
        for (int i = 0; i < 128; ++i) {
            matrix_intermediate_A[i] = torch::roll(matrix_intermediate_A[i], -i, 0);
            matrix_intermediate_B[i] = torch::roll(matrix_intermediate_B[i], -i, 0);
        }

        auto ct1 = CKKSEncrypt(flatten_pack(matrix1A, matrix1B), ckks_evaluator);
        auto ct2 = CKKSEncrypt(flatten_pack(matrix2A, matrix2B), ckks_evaluator);
        auto ct3 = CKKSEncrypt(flatten_pack(matrix3), ckks_evaluator);

        PhantomCiphertext res1, res2;
        mme.matrix_mul_ct128x64_ct128x64_transpose_gt(ct1, ct2, res1);
        CHECK(res1.chain_index() == ct1.chain_index() + 1);
        auto mm_res1 = tensor_from_vector(CKKSDecrypt(res1, ckks_evaluator), {2, 128, 128});

        REQUIRE(torch::allclose(mm_res1[0], matrix_intermediate_A, MAX_RTOL, MAX_ATOL));
        REQUIRE(torch::allclose(mm_res1[1], matrix_intermediate_B, MAX_RTOL, MAX_ATOL));

        mme.matrix_mul_ct128x128_ct128x128_gt(res1, ct3, res2);
        CHECK(res2.chain_index() == res1.chain_index() + 1);
        auto mm_res2 = tensor_from_vector(CKKSDecrypt(res2, ckks_evaluator), {2, 128, 128});

        REQUIRE(torch::allclose(mm_res2[0].slice(1, 0, 64), gt_matrix_A, MAX_RTOL, MAX_ATOL));
        REQUIRE(torch::allclose(mm_res2[1].slice(1, 64, 128), gt_matrix_B, MAX_RTOL, MAX_ATOL));

        torch::cuda::synchronize();
        BENCHMARK("matmul1") {
            mme.matrix_mul_ct128x64_ct128x64_transpose_gt(ct1, ct2, res1);  
            torch::cuda::synchronize();
        };
        
        torch::cuda::synchronize();
        BENCHMARK("matmul2") {
            mme.matrix_mul_ct128x128_ct128x128_gt(res1, ct3, res2);
            torch::cuda::synchronize();
        };
    }

    /*
    ct = np.random.randn(128, 768)
    pt = np.random.randn(768, 128)
    bias = np.random.randn(128)

    cts = row_pack_128x768(ct)
    pts = row_pack_768x64_type2(pt[:, :64], pt[:, 64:])

    res = multiply_ct128x768_pt768x128_type2(cts, pts)
    assert np.isclose(res[:SLOTS//2], (ct @ pt).flatten()).all()
    assert np.isclose(res[SLOTS//2:], (ct @ pt).flatten()).all()

    res_plus_bias = add_vec(res, row_pack_64x1_type2(bias[:64], bias[64:]))
    assert np.isclose(res_plus_bias[:SLOTS//2], (ct @ pt + bias).flatten()).all()
    assert np.isclose(res_plus_bias[SLOTS//2:], (ct @ pt + bias).flatten()).all()
    */
    SECTION("ct 128x768 pt 768x128") {
        torch::Tensor matrix_A = torch::randn({128, 768}, torch::kDouble);
        torch::Tensor matrix_B = torch::randn({768, 128}, torch::kDouble);
        torch::Tensor bias = torch::randn({128}, torch::kDouble);
        torch::Tensor matrix_C = torch::mm(matrix_A, matrix_B);

        auto packed_A = row_pack_128x768(matrix_A);
        vector<PhantomCiphertext> cts{
            CKKSEncrypt(packed_A[0], ckks_evaluator), 
            CKKSEncrypt(packed_A[1], ckks_evaluator), 
            CKKSEncrypt(packed_A[2], ckks_evaluator), 
        };
        auto pts = row_pack_768x128(matrix_B);
        auto bias_packed = row_pack_128x1(bias);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x768_pt768x128(cts, pts, res);
        CHECK(res.chain_index() == cts[0].chain_index() + 1);
        auto mm_res = tensor_from_vector(CKKSDecrypt(res, ckks_evaluator), {2, 128, 128});
        CHECK(torch::allclose(mm_res.index({0}), matrix_C, MAX_RTOL, MAX_ATOL));
        CHECK(torch::allclose(mm_res.index({1}), matrix_C, MAX_RTOL, MAX_ATOL));

        auto bias_pt = CKKSEncode(bias_packed, ckks_evaluator, &res);
        ckks_evaluator->evaluator.add_plain_inplace(res, bias_pt);
        mm_res = tensor_from_vector(CKKSDecrypt(res, ckks_evaluator), {2, 128, 128});
        CHECK(torch::allclose(mm_res.index({0}), matrix_C + bias.unsqueeze(0), MAX_RTOL, MAX_ATOL));
        CHECK(torch::allclose(mm_res.index({1}), matrix_C + bias.unsqueeze(0), MAX_RTOL, MAX_ATOL));

        torch::cuda::synchronize();
        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x768_pt768x128(cts, pts, res);
            torch::cuda::synchronize();
        };
    }

    /*
    ct = np.random.randn(128, 768)
    pt1 = np.random.randn(768, 64)
    pt2 = np.random.randn(768, 64)
    bias1 = np.random.randn(64)
    bias2 = np.random.randn(64)
    
    ct_packed = row_pack_128x768(ct)
    pt_packed = row_pack_768x64(pt1, pt2)
    bias_packed = row_pack_64x1(bias1, bias2)

    res = multiply_ct128x768_pt768x128(ct_packed, pt_packed)
    assert np.isclose(res[:SLOTS//2].reshape((128, 128))[:, :64], ct @ pt1).all()
    assert np.isclose(res[SLOTS//2:].reshape((128, 128))[:, :64], ct @ pt2).all()

    res = add_vec(res, bias_packed)
    assert np.isclose(res[:SLOTS//2].reshape((128, 128))[:, :64], ct @ pt1 + bias1).all()
    assert np.isclose(res[SLOTS//2:].reshape((128, 128))[:, :64], ct @ pt2 + bias2).all()
    */
    SECTION("ct 128x768 pt 768x64x2") {
        torch::Tensor matrix_A = torch::rand({128, 768}, torch::kDouble) - 0.5;
        torch::Tensor matrix_B1 = torch::rand({768, 64}, torch::kDouble) - 0.5;
        torch::Tensor matrix_B2 = torch::rand({768, 64}, torch::kDouble) - 0.5;
        torch::Tensor bias1 = torch::rand({64}, torch::kDouble) - 0.5;
        torch::Tensor bias2 = torch::rand({64}, torch::kDouble) - 0.5;
        torch::Tensor matrix_C1 = torch::mm(matrix_A, matrix_B1);
        torch::Tensor matrix_C2 = torch::mm(matrix_A, matrix_B2);
        vector<vector<double>> packed_A = row_pack_128x768(matrix_A);
        vector<PhantomCiphertext> cts{
            CKKSEncrypt(packed_A[0], ckks_evaluator), 
            CKKSEncrypt(packed_A[1], ckks_evaluator), 
            CKKSEncrypt(packed_A[2], ckks_evaluator), 
        };
        vector<vector<double>> pts = row_pack_768x64x2(matrix_B1, matrix_B2);
        auto bias_packed = row_pack_64x1x2(bias1, bias2);

        PhantomCiphertext res;
        mme.matrix_mul_ct128x768_pt768x64x2(cts, pts, res);
        CHECK(res.chain_index() == cts[0].chain_index() + 1);

        {
        torch::Tensor mm_res = tensor_from_vector(CKKSDecrypt(res, ckks_evaluator), {2, 128, 128});
        torch::Tensor mm_res1 = mm_res.index({0}).slice(1, 0, 64).clone();
        torch::Tensor mm_res2 = mm_res.index({1}).slice(1, 0, 64).clone();

        REQUIRE(torch::allclose(matrix_C1, mm_res1, MAX_RTOL, MAX_ATOL));
        REQUIRE(torch::allclose(matrix_C2, mm_res2, MAX_RTOL, MAX_ATOL));
        }

        auto bias_pt = CKKSEncode(bias_packed, ckks_evaluator, &res);
        ckks_evaluator->evaluator.add_plain_inplace(res, bias_pt);

        {
        auto mm_res = CKKSDecrypt(res, ckks_evaluator);
        torch::Tensor mm_res1 = torch::from_blob(mm_res.data(), {128, 128}, torch::kDouble).slice(1, 0, 64);
        torch::Tensor mm_res2 = torch::from_blob(mm_res.data() + 128 * 128, {128, 128}, torch::kDouble).slice(1, 0, 64);

        matrix_C1 += bias1.unsqueeze(0);
        matrix_C2 += bias2.unsqueeze(0);
        REQUIRE(torch::allclose(matrix_C1, mm_res1, MAX_RTOL, MAX_ATOL));
        REQUIRE(torch::allclose(matrix_C2, mm_res2, MAX_RTOL, MAX_ATOL));
        }

        torch::cuda::synchronize();
        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x768_pt768x64x2(cts, pts, res);
            torch::cuda::synchronize();
        };
    }

    /*
    ct = np.random.randn(128, 768)
    pt = np.random.randn(768, 768)
    bias = np.random.randn(768)
    
    ct_packed = row_pack_128x768(ct)
    pt_packed = row_pack_768x768(pt)
    bias_packed = row_pack_768x1(bias)

    gt = ct @ pt

    res = multiply_ct128x768_pt768x768(ct_packed, pt_packed)
    for i in range(3):
        assert np.isclose(res[i][:SLOTS//2].reshape((128, 128)), gt[:, 256*i: 256*i+128]).all()
        assert np.isclose(res[i][SLOTS//2:].reshape((128, 128)), gt[:, 256*i+128: 256*i+256]).all()

    res = add_vec(res, bias_packed)
    gt = ct @ pt + bias
    for i in range(3):
        assert np.isclose(res[i][:SLOTS//2].reshape((128, 128)), gt[:, 256*i: 256*i+128]).all()
        assert np.isclose(res[i][SLOTS//2:].reshape((128, 128)), gt[:, 256*i+128: 256*i+256]).all()
    */
    SECTION("ct 128x768 pt 768x768") {
        torch::Tensor matrix_A = torch::randn({128, 768}, torch::kDouble);
        torch::Tensor matrix_B = torch::randn({768, 768}, torch::kDouble);
        torch::Tensor bias = torch::randn({768}, torch::kDouble);
        torch::Tensor matrix_C = torch::mm(matrix_A, matrix_B);
        vector<vector<double>> packed_A = row_pack_128x768(matrix_A);
        vector<PhantomCiphertext> cts{
            CKKSEncrypt(packed_A[0], ckks_evaluator),
            CKKSEncrypt(packed_A[1], ckks_evaluator),
            CKKSEncrypt(packed_A[2], ckks_evaluator),
        };
        auto pts = row_pack_768x768(matrix_B);
        auto bias_packed = row_pack_768x1(bias);

        vector<PhantomCiphertext> res;
        mme.matrix_mul_ct128x768_pt768x768(cts, pts, res);
        for (int i=0; i<3; i++) {
            CHECK(res[i].chain_index() == cts[i].chain_index() + 1);
        }

        torch::Tensor mm_result = torch::zeros({128, 768}, torch::kDouble);
        for (int i = 0; i < 3; i++) {
            auto mm_res = CKKSDecrypt(res[i], ckks_evaluator);
            mm_result.slice(1, i * 256, i * 256 + 128) = torch::from_blob(mm_res.data(), {128, 128}, torch::kDouble).clone();
            mm_result.slice(1, i * 256 + 128, i * 256 + 256) = torch::from_blob(mm_res.data() + 128 * 128, {128, 128}, torch::kDouble).clone();
        }

        REQUIRE(torch::allclose(mm_result, matrix_C, MAX_RTOL, MAX_ATOL));

        for (int i=0; i<res.size(); i++) {
            auto bias_pt = CKKSEncode(bias_packed[i], ckks_evaluator, &res[i]);
            ckks_evaluator->evaluator.add_plain_inplace(res[i], bias_pt);
        }

        for (int i=0; i<3; i++) {
            auto mm_res = CKKSDecrypt(res[i], ckks_evaluator);
            mm_result.slice(1, i * 256, i * 256 + 128) = torch::from_blob(mm_res.data(), {128, 128}, torch::kDouble).clone();
            mm_result.slice(1, i * 256 + 128, i * 256 + 256) = torch::from_blob(mm_res.data() + 128 * 128, {128, 128}, torch::kDouble).clone();
        }
        REQUIRE(torch::allclose(mm_result, matrix_C + bias.unsqueeze(0), MAX_RTOL, MAX_ATOL));

        torch::cuda::synchronize();
        BENCHMARK("matmul") {
            mme.matrix_mul_ct128x768_pt768x768(cts, pts, res);
            torch::cuda::synchronize();
        };
    }

}