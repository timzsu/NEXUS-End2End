#include "bert/bert.cuh"
#include "nn/nexus_utility.cuh"

namespace nexus {

void BertLayer::load_state_dict(torch::Dict<torch::IValue, torch::IValue>& state_dict, std::string prefix) {
    self_attention.load_state_dict(state_dict, prefix + ".attention");
    ln1.load_state_dict(state_dict, prefix + ".attention.output.LayerNorm");
    mlp.load_state_dict(state_dict, prefix);
    ln2.load_state_dict(state_dict, prefix + ".output.LayerNorm");
}

void BertLayer::pack_weights() {
    self_attention.pack_weights();
    ln1.pack_weights();
    mlp.pack_weights();
    ln2.pack_weights();
}

std::vector<PhantomCiphertext> BertLayer::forward(vector<PhantomCiphertext>& x, FlatVec attention_mask) {

    auto attn_output = self_attention.forward(x, attention_mask);
    torch::cuda::synchronize();
    std::cout << "Attention Finished" << std::endl;

    layer_norm1_timer.start();
    bootstrap(attn_output, bootstrapper);
    for (auto& ct: attn_output)
      ckks->evaluator.mod_switch_to_inplace(ct, chain_idx(6));
    auto attn_output_normalized = ln1.forward(attn_output);
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
    auto mlp_output_normalized = ln2.forward(mlp_output);
    bootstrap(mlp_output_normalized, bootstrapper);
    torch::cuda::synchronize();
    layer_norm2_timer.stop();
    std::cout << "LN2 Finished in " << layer_norm2_timer.duration() / 1e3 << " seconds. " << std::endl;

    return mlp_output_normalized;
}

torch::Tensor BertLayer::forward(torch::Tensor x, torch::Tensor attention_mask) {
  auto attn_output = self_attention.forward(x, attention_mask);
  attn_output = ln1.forward(attn_output);
  auto mlp_output = mlp.forward(attn_output);
  mlp_output = ln2.forward(mlp_output);
  return mlp_output;
}

}