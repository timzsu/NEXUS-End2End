#pragma once

#include "Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "nn/row_pack.h"
#include "phantom.h"

namespace nexus {
using namespace std;
using namespace phantom;

class SoftmaxEvaluator {
 private:
  std::shared_ptr<CKKSEvaluator> ckks;
  std::shared_ptr<Bootstrapper> bootstrapper;

 public:
  SoftmaxEvaluator(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper) : ckks(ckks), bootstrapper(bootstrapper) {}

  void softmax(PhantomCiphertext &x, PhantomCiphertext &res, int len);
  void softmax_128x128(PhantomCiphertext &x, PhantomCiphertext &res, FlatVec attention_mask);
};
}  // namespace nexus
