#include "bert/bert.cuh"
#include "nn/nexus_utility.cuh"

namespace nexus {

void BertLayer::pack_weights() {
    self_attention.pack_weights();
    mlp.pack_weights();
}

void BertLayer::bootstrap(PhantomCiphertext &x) {
  while (x.coeff_modulus_size() > 1) {
    ckks->evaluator.mod_switch_to_next_inplace(x);
  }
  PhantomCiphertext rtn;
  bootstrapper->set_final_scale(x.scale());
  bootstrapper->bootstrap_3(rtn, x);
  x = rtn;
}

void BertLayer::bootstrap_insecure(std::vector<PhantomCiphertext>& ct_vec) {
  for (auto& ct: ct_vec) {
    auto pt = CKKSDecrypt(ct, ckks);
    ct = CKKSEncrypt(pt, ckks);
  }
}

std::vector<PhantomCiphertext> BertLayer::forward(vector<PhantomCiphertext>& x) {
    auto attn_output = self_attention.forward(x);
    bootstrap_insecure(attn_output);
    std::vector<PhantomCiphertext> attn_output_normalized;
    ln_evaluator.layer_norm_128x768(attn_output, attn_output_normalized);
    bootstrap_insecure(attn_output_normalized);
    auto mlp_output = mlp.forward(attn_output_normalized);
    bootstrap_insecure(mlp_output);
    std::vector<PhantomCiphertext> mlp_output_normalized;
    ln_evaluator.layer_norm_128x768(mlp_output, mlp_output_normalized);
    bootstrap_insecure(mlp_output_normalized);
    return mlp_output_normalized;
}

torch::Tensor BertLayer::forward(torch::Tensor x) {
  auto attn_output = self_attention.forward(x);
  attn_output = torch::layer_norm(attn_output, 768);
  auto mlp_output = mlp.forward(attn_output);
  mlp_output = torch::layer_norm(mlp_output, 768);
  return mlp_output;
}

}