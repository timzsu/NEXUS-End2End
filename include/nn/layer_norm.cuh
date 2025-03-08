#pragma once

#include "Bootstrapper.cuh"
#include "ckks_evaluator.cuh"

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
  void layer_norm_128x768(std::vector<PhantomCiphertext> &x, std::vector<PhantomCiphertext> &res);
};
}  // namespace nexus
