#include "bert/bert.cuh"

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

std::vector<PhantomCiphertext> BertLayer::forward(vector<PhantomCiphertext>& x) {
    auto attn_output = self_attention.forward(x);
    for (auto& c: attn_output) {
        bootstrap(c);
        ln_evaluator.layer_norm(c, c, 1024);
        bootstrap(c);
    }
    auto mlp_output = mlp.forward(attn_output);
    for (auto& c: attn_output) {
        bootstrap(c);
        ln_evaluator.layer_norm(c, c, 1024);
        bootstrap(c);
    }
    return mlp_output;
}

}