#include "bert/bert.cuh"
#include "nn/nexus_utility.cuh"
#include "nn/params.cuh"

#include <precompiled/catch2_includes.h>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

constexpr double MAX_RTOL=1e-3;
constexpr double MAX_ATOL=1e-2;

torch::Tensor random_tensor(torch::IntArrayRef size, double min, double max) {
    return torch::rand(size, torch::kDouble) * (max - min) + min;   
}

TEST_CASE("BERT Components") {
    auto ckks_evaluator = setup();

    SECTION("Attention") {
        BertAttention attention(ckks_evaluator);

        torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
        auto gt_output = attention.forward(input.to(torch::kFloat));

        auto packed_input = row_pack_128x768(input);
        std::vector<PhantomCiphertext> input_ct;
        for (auto &inp : packed_input) {
            input_ct.push_back(CKKSEncrypt(inp, ckks_evaluator));
        }

        attention.pack_weights();

        torch::cuda::synchronize();
        BENCHMARK("forward") {
            std::vector<PhantomCiphertext> res, input_copy = input_ct;
            auto out = attention.forward(input_ct);
            torch::cuda::synchronize();
        };
        auto out = attention.forward(input_ct);

        torch::Tensor attn_output = tensor_from_ciphertexts(out, ckks_evaluator);

        CHECK(torch::allclose(attn_output.to(torch::kFloat), gt_output, MAX_RTOL, MAX_ATOL));
    }
    
    SECTION("MLP") {
        BertMLP mlp(ckks_evaluator);

        torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
        torch::Tensor gt_output = mlp.forward(input.to(torch::kFloat));

        auto packed_input = row_pack_128x768(input);
        std::vector<PhantomCiphertext> input_ct;
        for (auto &inp : packed_input) {
            input_ct.push_back(CKKSEncrypt(inp, ckks_evaluator));
        }

        mlp.pack_weights();

        torch::cuda::synchronize();
        BENCHMARK("forward") {
            std::vector<PhantomCiphertext> res, input_copy = input_ct;
            auto out = mlp.forward(input_ct);
            torch::cuda::synchronize();
        };
        auto out = mlp.forward(input_ct);

        torch::Tensor output = tensor_from_ciphertexts(out, ckks_evaluator);

        CHECK(torch::allclose(output.to(torch::kFloat), gt_output, MAX_RTOL, MAX_ATOL));
    }
}

TEST_CASE("BERT Layer") {

    auto [ckks_evaluator, bootstrapper] = setup<true>();

    BertLayer bert_layer(ckks_evaluator, bootstrapper);

    torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
    auto gt_output = bert_layer.forward(input.to(torch::kFloat));

    auto packed_input = row_pack_128x768(input);
    std::vector<PhantomCiphertext> input_ct;
    for (auto &inp : packed_input) {
        input_ct.push_back(CKKSEncrypt(inp, ckks_evaluator));
    }

    bert_layer.pack_weights();

    torch::cuda::synchronize();
    BENCHMARK("forward") {
        std::vector<PhantomCiphertext> res, input_copy = input_ct;
        auto out = bert_layer.forward(input_ct);
        torch::cuda::synchronize();
    };
    auto out = bert_layer.forward(input_ct);

    torch::Tensor output = tensor_from_ciphertexts(out, ckks_evaluator);

    CHECK(torch::allclose(output.to(torch::kFloat), gt_output, MAX_RTOL, MAX_ATOL));
}