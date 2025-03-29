#include "nn/nexus_utility.cuh"

#include "nn/softmax.cuh"
#include "nn/gelu.cuh"
#include "nn/layer_norm.cuh"
#include "nn/argmax.cuh"
#include "nn/params.cuh"

#include <precompiled/catch2_includes.h>
#include <precompiled/torch_includes.h>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;


// size_t N = 1ULL << 16;
// double SCALE = pow(2.0, 40);
// size_t L = 21;
constexpr double MAX_RTOL=1e-3;
constexpr double MAX_ATOL=1e-2;

torch::Tensor random_tensor(torch::IntArrayRef size, double min, double max) {
    return torch::rand(size, torch::kDouble) * (max - min) + min;   
}

TEST_CASE("Non-linear Operations") {
    
    auto [ckks_evaluator, bootstrapper] = setup<1>();

    SECTION("Softmax") {
        SoftmaxEvaluator softmax_evaluator(ckks_evaluator, bootstrapper);

        torch::Tensor matrix_A = random_tensor({128, 128}, -1, 1);
        torch::Tensor matrix_B = random_tensor({128, 128}, -1, 1);
        matrix_A -= std::get<0>(matrix_A.max(1, true));

        torch::Tensor attention_mask = torch::randint(0, 2, {128, 128}, torch::kBool);
        auto mask = convert_mask(attention_mask);

        matrix_A.masked_fill_(~attention_mask, -100);
        matrix_B.masked_fill_(~attention_mask, -100);
        
        torch::Tensor matrix_res_A = torch::softmax(matrix_A, 1);
        torch::Tensor matrix_res_B = torch::softmax(matrix_B, 1);
        torch::Tensor gt_output = torch::stack({matrix_res_A, matrix_res_B});

        PhantomCiphertext ct_matrix = CKKSEncrypt(flatten_pack(matrix_A, matrix_B), ckks_evaluator);
        torch::cuda::synchronize();
        BENCHMARK("softmax") {
            PhantomCiphertext res, input_copy = ct_matrix;
            softmax_evaluator.softmax_128x128(ct_matrix, res, mask);
            torch::cuda::synchronize();
        };
        PhantomCiphertext res;
        softmax_evaluator.softmax_128x128(ct_matrix, res, mask);
        CHECK(res.chain_index() == ct_matrix.chain_index() + 9);
        auto mm_res = CKKSDecrypt(res, ckks_evaluator);
        torch::Tensor tensor_res = tensor_from_vector(mm_res, {2, 128, 128});
        
        cout << std::format(
            "gt_output's value ranges from {:.4f} to {:.4f}, with average abs value {:.4f}. The maximum absolute difference is {:.4f}. ", 
            gt_output.min().item<double>(), 
            gt_output.max().item<double>(), 
            gt_output.abs().mean().item<double>(), 
            (tensor_res - gt_output).abs().max().item<double>()
        ) << endl;

        REQUIRE(torch::allclose(tensor_res, gt_output, MAX_RTOL, MAX_ATOL));
    }

    SECTION("GELU") {
        GELUEvaluator gelu_evaluator(ckks_evaluator, bootstrapper);

        torch::Tensor matrix_A = torch::randn({128, 256}, torch::kDouble);
        torch::Tensor matrix_res = torch::nn::functional::gelu(matrix_A);
        PhantomCiphertext ct_matrix = CKKSEncrypt(vector_from_tensor(matrix_A), ckks_evaluator);

        torch::cuda::synchronize();
        BENCHMARK("gelu") {
            PhantomCiphertext res, input_copy = ct_matrix;
            gelu_evaluator.gelu(input_copy, res);
            torch::cuda::synchronize();
        };
        PhantomCiphertext res;
        auto original_chain_index = ct_matrix.chain_index();
        gelu_evaluator.gelu(ct_matrix, res);
        CHECK(res.chain_index() == original_chain_index + 6);
        auto mm_res = CKKSDecrypt(res, ckks_evaluator);
        torch::Tensor tensor_res = tensor_from_vector(mm_res, {128, 256});
        
        cout << std::format(
            "gt_output's value ranges from {:.4f} to {:.4f}, with average abs value {:.4f}. The maximum absolute difference is {:.4f}. ", 
            matrix_res.min().item<double>(), 
            matrix_res.max().item<double>(), 
            matrix_res.abs().mean().item<double>(), 
            (tensor_res - matrix_res).abs().max().item<double>()
        ) << endl;

        REQUIRE(torch::allclose(matrix_res, tensor_res, MAX_RTOL, MAX_ATOL));

    }

    SECTION("Layer Norm") {
        LNEvaluator ln_evaluator(ckks_evaluator, bootstrapper);

        torch::Tensor matrix_A = random_tensor({16, 768}, -3, 3);
        matrix_A -= matrix_A.mean(1, true);
        torch::Tensor placeholder = torch::zeros({16, 2048-768}, torch::kDouble);
        torch::Tensor matrix_res = torch::layer_norm(matrix_A, 768);
        PhantomCiphertext ct_matrix = CKKSEncrypt(vector_from_tensor(torch::concat({matrix_A, placeholder}, 1)), ckks_evaluator);

        torch::cuda::synchronize();
        BENCHMARK("layer_norm") {
            PhantomCiphertext res, input_copy = ct_matrix;
            ln_evaluator.layer_norm(input_copy, res, 1024);
            torch::cuda::synchronize();
        };
        PhantomCiphertext res;
        auto original_chain_index = ct_matrix.chain_index();
        ln_evaluator.layer_norm(ct_matrix, res, 1024);
        CHECK(res.chain_index() == original_chain_index + 18);
        auto mm_res = CKKSDecrypt(res, ckks_evaluator);
        torch::Tensor tensor_res = tensor_from_vector(mm_res, {16, 2048});

        REQUIRE(torch::allclose(tensor_res.slice(1, 0, 768), matrix_res, MAX_RTOL, MAX_ATOL));

    }

    SECTION("Layer Norm 128x768") {
        LNEvaluator ln_evaluator(ckks_evaluator, bootstrapper);

        torch::Tensor matrix_A = random_tensor({128, 768}, -3, 3);
        torch::Tensor matrix_res = torch::layer_norm(matrix_A, 768);
        auto packed_A = row_pack_128x768(matrix_A);
        vector<PhantomCiphertext> ct_matrix{
            CKKSEncrypt(packed_A[0], ckks_evaluator, chain_idx(6)),
            CKKSEncrypt(packed_A[1], ckks_evaluator, chain_idx(6)),
            CKKSEncrypt(packed_A[2], ckks_evaluator, chain_idx(6)),
        };

        torch::cuda::synchronize();
        BENCHMARK("layer_norm") {
            std::vector<PhantomCiphertext> res;
            ln_evaluator.layer_norm_128x768(ct_matrix, res);
            torch::cuda::synchronize();
        };
        std::vector<PhantomCiphertext> res;
        ln_evaluator.layer_norm_128x768(ct_matrix, res);
        CHECK(res[0].chain_index() == ct_matrix[0].chain_index() + 16);

        torch::Tensor output = tensor_from_ciphertexts(res, ckks_evaluator);
        
        cout << std::format(
            "gt_output's value ranges from {:.4f} to {:.4f}, with average abs value {:.4f}. The maximum absolute difference is {:.4f}. ", 
            matrix_res.min().item<double>(), 
            matrix_res.max().item<double>(), 
            matrix_res.abs().mean().item<double>(), 
            (output - matrix_res).abs().max().item<double>()
        ) << endl;

        REQUIRE(torch::allclose(output, matrix_res, MAX_RTOL, MAX_ATOL));

    }

    SECTION("Argmax") {
        ArgmaxEvaluator argmax_evaluator(ckks_evaluator, bootstrapper, L);
        PhantomCiphertext cipher_input;
        PhantomCiphertext cipher_output;
    
        int argmax_input_size = 8; // FIXME: Larger size will fail
        torch::Tensor input_tensor = torch::zeros({slot_count}, torch::kDouble);
        input_tensor.slice(0, 0, argmax_input_size) = random_tensor({argmax_input_size}, -0.1, 0.1);
        torch::Tensor gt = torch::zeros({argmax_input_size}, torch::kDouble);
        gt[torch::argmax(input_tensor.slice(0, 0, argmax_input_size))] = 1.0;
    
        cipher_input = CKKSEncrypt(vector_from_tensor(input_tensor), ckks_evaluator);

        argmax_evaluator.argmax(cipher_input, cipher_output, argmax_input_size);
        auto mm_res = CKKSDecrypt(cipher_output, ckks_evaluator);
        torch::Tensor pred = tensor_from_vector(mm_res, {argmax_input_size});

        cout << gt << endl << pred << endl;
        REQUIRE(torch::allclose(gt, pred, MAX_RTOL, MAX_ATOL));

        BENCHMARK("argmax") {
            argmax_evaluator.argmax(cipher_input, cipher_output, argmax_input_size);
        };
    }
}