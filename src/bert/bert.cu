#include "bert/bert.cuh"
#include "nn/nexus_utility.cuh"

namespace nexus {

void BertLayer::pack_weights() {
    self_attention.pack_weights();
    mlp.pack_weights();
}

std::vector<PhantomCiphertext> BertLayer::forward(vector<PhantomCiphertext>& x, FlatVec attention_mask) {

    auto attn_output = self_attention.forward(x, attention_mask);
    torch::cuda::synchronize();
    std::cout << "Attention Finished" << std::endl;

    layer_norm1_timer.start();
    bootstrap(attn_output, bootstrapper);
    std::vector<PhantomCiphertext> attn_output_normalized;
    ln_evaluator.layer_norm_128x768(attn_output, attn_output_normalized);
    bootstrap(attn_output_normalized, bootstrapper);
    for (auto& ct: attn_output_normalized) {
      ckks->evaluator.mod_switch_to_inplace(ct, chain_idx(14));
    }
    torch::cuda::synchronize();
    layer_norm1_timer.stop();
    std::cout << "LN1 Finished in " << layer_norm1_timer.duration() / 1e3 << " seconds. " << std::endl;

    auto mlp_output = mlp.forward(attn_output_normalized);
    torch::cuda::synchronize();
    std::cout << "MLP Finished" << std::endl;

    layer_norm2_timer.start();
    std::vector<PhantomCiphertext> mlp_output_normalized;
    ln_evaluator.layer_norm_128x768(mlp_output, mlp_output_normalized);
    bootstrap(mlp_output_normalized, bootstrapper);
    torch::cuda::synchronize();
    layer_norm2_timer.stop();
    std::cout << "LN2 Finished in " << layer_norm2_timer.duration() / 1e3 << " seconds. " << std::endl;

    return mlp_output_normalized;
}

torch::Tensor BertLayer::forward(torch::Tensor x, torch::Tensor attention_mask) {
  auto attn_output = self_attention.forward(x, attention_mask);
  attn_output = torch::layer_norm(attn_output, 768);
  auto mlp_output = mlp.forward(attn_output);
  mlp_output = torch::layer_norm(mlp_output, 768);
  return mlp_output;
}

}