#include "bert/bert.cuh"

namespace nexus {

void BertLayer::random_init() {
    self_attention.random_init();
    mlp.random_init();
}

std::vector<PhantomCiphertext> BertLayer::forward(vector<PhantomCiphertext>& x) {
    auto attn_output = self_attention.forward(x);
    for (auto& c: attn_output) {
        ln_evaluator.layer_norm(c, c, 128);
    }
    auto mlp_output = mlp.forward(attn_output);
    for (auto& c: attn_output) {
        ln_evaluator.layer_norm(c, c, 128);
    }
    return mlp_output;
}

}