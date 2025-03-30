#pragma once

#include "Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "nn/row_pack.h"

#include <precompiled/torch_includes.h>
namespace nexus {
using namespace std;
using namespace phantom;

class LNEvaluator {
 private:
  std::shared_ptr<CKKSEvaluator> ckks;
  std::shared_ptr<Bootstrapper> bootstrapper;

 public:
  LNEvaluator(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper) : ckks(ckks), bootstrapper(bootstrapper) {}
  void layer_norm(PhantomCiphertext &x, PhantomCiphertext &res, int len);
  void layer_norm_128x768(std::vector<PhantomCiphertext> &x, std::vector<PhantomCiphertext> &res, const FlatVecArray& weight, const FlatVecArray& bias);
};

class LayerNorm: public torch::nn::Module {
  private: 
    LNEvaluator ln_evaluator;
    torch::Tensor weight, bias;
    Timer layer_norm_timer;

    FlatVecArray weight_packed, bias_packed;

  public:
    LayerNorm(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper) : ln_evaluator(ckks, bootstrapper) {
      weight = torch::ones(768);
      bias = torch::zeros(768);
      torch::nn::init::uniform_(weight, 0.5, 1.5);
      torch::nn::init::uniform_(bias, -0.5, 0.5);
    }

    void load_state_dict(torch::Dict<torch::IValue, torch::IValue>& state_dict, std::string prefix="") {
      weight.copy_(state_dict.at(prefix + ".weight").toTensor());
      bias.copy_(state_dict.at(prefix + ".bias").toTensor());
    }

    void pack_weights() {
      std::tie(weight_packed, bias_packed) = row_pack_layer_norm(weight.to(torch::kDouble), bias.to(torch::kDouble));
    }

    std::vector<PhantomCiphertext> forward(std::vector<PhantomCiphertext>& x) {
      layer_norm_timer.start();
      std::vector<PhantomCiphertext> res(3);
      ln_evaluator.layer_norm_128x768(x, res, weight_packed, bias_packed);
      layer_norm_timer.stop();
      return res;
    }

    torch::Tensor forward(torch::Tensor x) {
      return torch::layer_norm(x, 768, weight, bias);
    }

    void print_time() {
      cout << "LayerNorm takes " << layer_norm_timer.duration() << "ms" << endl;
    }
};

}  // namespace nexus
