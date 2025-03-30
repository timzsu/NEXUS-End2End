#include "bert/bert.cuh"
#include "nn/nexus_utility.cuh"
#include "nn/params.cuh"

#include <precompiled/catch2_includes.h>

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

constexpr double MAX_RTOL=5e-2;
constexpr double MAX_ATOL=0.5;

torch::Tensor random_tensor(torch::IntArrayRef size, double min, double max) {
    return torch::rand(size, torch::kDouble) * (max - min) + min;   
}

TEST_CASE("BERT Components") {
    auto [ckks_evaluator, bootstrapper] = setup<1>();

    std::ifstream file("/cephfs/suzhengyuan/secure_quantization/tanmay/quad_2quad_COLA/state_dict.pt", std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto state_dict = torch::pickle_load(data).toGenericDict();

    SECTION("Attention") {
        BertAttention attention(ckks_evaluator, bootstrapper);
        attention.load_state_dict(state_dict, "bert.encoder.layer.0.attention");

        torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
        torch::Tensor attention_mask = torch::ones({128, 128}, torch::kBool);
        attention_mask.slice(1, 126) = 0;
        auto mask = convert_mask(attention_mask);
        auto gt_output = attention.forward(input.to(torch::kFloat), attention_mask);

        auto packed_input = row_pack_128x768(input);
        std::vector<PhantomCiphertext> input_ct;
        for (auto &inp : packed_input) {
            input_ct.push_back(CKKSEncrypt(inp, ckks_evaluator));
        }

        attention.pack_weights();

        torch::cuda::synchronize();
        BENCHMARK("forward") {
            std::vector<PhantomCiphertext> res, input_copy = input_ct;
            auto out = attention.forward(input_ct, mask);
            torch::cuda::synchronize();
        };
        auto out = attention.forward(input_ct, mask);
        attention.print_time();

        torch::Tensor attn_output = tensor_from_ciphertexts(out, ckks_evaluator);

        cout << std::format(
            "gt_output's value ranges from {:.4f} to {:.4f}, with average abs value {:.4f}. The maximum absolute difference is {:.4f}. ", 
            gt_output.min().item<double>(), 
            gt_output.max().item<double>(), 
            gt_output.abs().mean().item<double>(), 
            (attn_output - gt_output).abs().max().item<double>()
        ) << endl;

        cout << (attn_output.to(torch::kFloat) - gt_output).abs().max() << endl;

        CHECK(torch::allclose(attn_output.to(torch::kFloat), gt_output, MAX_RTOL, MAX_ATOL));
    }
    
    SECTION("MLP") {
        BertMLP mlp(ckks_evaluator, bootstrapper);
        mlp.load_state_dict(state_dict, "bert.encoder.layer.0");

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
        mlp.print_time();

        torch::Tensor output = tensor_from_ciphertexts(out, ckks_evaluator);

        cout << std::format(
            "gt_output's value ranges from {:.4f} to {:.4f}, with average abs value {:.4f}. The maximum absolute difference is {:.4f}. ", 
            gt_output.min().item<double>(), 
            gt_output.max().item<double>(), 
            gt_output.abs().mean().item<double>(), 
            (output - gt_output).abs().max().item<double>()
        ) << endl;

        CHECK(torch::allclose(output.to(torch::kFloat), gt_output, MAX_RTOL, MAX_ATOL));
    }
}

TEST_CASE("BERT Layer") {

    auto [ckks_evaluator, bootstrapper] = setup<true>();

    std::ifstream file("/cephfs/suzhengyuan/secure_quantization/tanmay/quad_2quad_COLA/state_dict.pt", std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    auto state_dict = torch::pickle_load(data).toGenericDict();

    BertLayer bert_layer(ckks_evaluator, bootstrapper);
    bert_layer.load_state_dict(state_dict, "bert.encoder.layer.0");
    bert_layer.pack_weights();

    SECTION("Faithful Execution") {
        torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
        torch::Tensor attention_mask = torch::ones({128, 128}, torch::kBool);
        // attention_mask.slice(1, 126) = 0;
        auto mask = convert_mask(attention_mask);
        auto gt_output = bert_layer.forward(input.to(torch::kFloat), attention_mask);

        auto packed_input = row_pack_128x768(input);
        std::vector<PhantomCiphertext> input_ct;
        for (auto &inp : packed_input) {
            input_ct.push_back(CKKSEncrypt(inp, ckks_evaluator));
        }

        torch::cuda::synchronize();
        BENCHMARK("forward") {
            std::vector<PhantomCiphertext> res, input_copy = input_ct;
            auto out = bert_layer.forward(input_ct, mask);
            torch::cuda::synchronize();
        };
        Timer timer;
        auto out = bert_layer.forward(input_ct, mask);
        torch::cuda::synchronize();
        timer.stop("End to end run time (ms): ");
        bert_layer.print_time();

        torch::Tensor output = tensor_from_ciphertexts(out, ckks_evaluator);

        cout << std::format(
            "gt_output's value ranges from {:.4f} to {:.4f}, with average abs value {:.4f}. The maximum absolute difference is {:.4f}. ", 
            gt_output.min().item<double>(), 
            gt_output.max().item<double>(), 
            gt_output.abs().mean().item<double>(), 
            (output - gt_output).abs().max().item<double>()
        ) << endl;
        
        CHECK(torch::allclose(output.to(torch::kFloat), gt_output, MAX_RTOL, MAX_ATOL));
    }
}


TEST_CASE("BERT Encoder") {

    auto [ckks_evaluator, bootstrapper] = setup<true>();

    BertEncoder bert_encoder(12, ckks_evaluator, bootstrapper);

    torch::Tensor input = random_tensor({128, 768}, -0.5, 0.5);
    torch::Tensor attention_mask = torch::ones({128, 128}, torch::kBool);
    // attention_mask.slice(1, 126) = 0;
    auto mask = convert_mask(attention_mask);
    auto gt_output = bert_encoder.forward(input.to(torch::kFloat), attention_mask);

    auto packed_input = row_pack_128x768(input);
    std::vector<PhantomCiphertext> input_ct;
    for (auto &inp : packed_input) {
        input_ct.push_back(CKKSEncrypt(inp, ckks_evaluator));
    }

    bert_encoder.pack_weights();

    torch::cuda::synchronize();
    BENCHMARK("forward") {
        std::vector<PhantomCiphertext> res, input_copy = input_ct;
        auto out = bert_encoder.forward(input_ct, mask);
        torch::cuda::synchronize();
    };
    Timer timer;
    auto out = bert_encoder.forward(input_ct, mask);
    torch::cuda::synchronize();
    timer.stop("End to end run time (ms): ");
    bert_encoder.print_time();

    torch::Tensor output = tensor_from_ciphertexts(out, ckks_evaluator);

    cout << std::format(
        "gt_output's value ranges from {:.4f} to {:.4f}, with average abs value {:.4f}. The maximum absolute difference is {:.4f}. ", 
        gt_output.min().item<double>(), 
        gt_output.max().item<double>(), 
        gt_output.abs().mean().item<double>(), 
        (output - gt_output).abs().max().item<double>()
    ) << endl;
    
    CHECK(torch::allclose(output.to(torch::kFloat), gt_output, MAX_RTOL, MAX_ATOL));
}