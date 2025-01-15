#include "nn/layer_norm.cuh"
#include "nn/nexus_utility.cuh"

using namespace nexus;

void LNEvaluator::layer_norm(PhantomCiphertext &a, PhantomCiphertext &y, int len) {
  PhantomCiphertext tmp, x2;

  int log_step = log2(len);
  ckks->evaluator.rotate_vector(a, -len, *ckks->galois_keys, tmp);
  ckks->evaluator.add_inplace(a, tmp);

  ckks->evaluator.square(a, x2);
  ckks->evaluator.relinearize_inplace(x2, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(x2);

  tmp = x2;
  for (int i = 0; i < log_step; ++i) {
    ckks->evaluator.rotate_vector(tmp, pow(2, i), *ckks->galois_keys, y);
    ckks->evaluator.add_inplace(y, tmp);
    tmp = y;
  }

  PhantomPlaintext delta;
  ckks->encoder.encode(1.0 / 768, y.params_id(), y.scale(), delta);
  ckks->evaluator.multiply_plain_inplace(y, delta);
  ckks->evaluator.rescale_to_next_inplace(y);

  y = ckks->invert_sqrt(y, 4, 2);

  ckks->evaluator.mod_switch_to_inplace(a, y.params_id());
  ckks->evaluator.multiply(y, a, y);
  ckks->evaluator.relinearize_inplace(y, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(y);

  // cout << "Moduli left after LayerNorm: " << y.coeff_modulus_size() << endl;
}

void LNEvaluator::layer_norm_128x768(std::vector<PhantomCiphertext> &x, std::vector<PhantomCiphertext> &res) {
  std::vector<PhantomCiphertext> tmp(3), z(3), y(3);
  res.resize(3);

  for (int i=0; i<3; i++) {
    tmp[i] = quick_sum(x[i], ckks, 128);
  }
  PhantomCiphertext sumx, rot_sumx;
  ckks->evaluator.add_many(tmp, sumx);
  ckks->evaluator.rotate_vector(sumx, slot_count / 2, *(ckks->galois_keys), rot_sumx);
  ckks->evaluator.add_inplace(sumx, rot_sumx);

  std::vector<double> const_1by768(slot_count, 1./768);
  std::vector<double> const_1(slot_count, 1.);
  ckks->evaluator.multiply_vector_inplace_reduced_error(sumx, const_1by768);
  ckks->evaluator.rescale_to_next_inplace(sumx);
  for (int i=0; i<3; i++) {
    PhantomCiphertext x_copy = x[i];
    ckks->evaluator.mod_switch_to_inplace(x_copy, sumx.chain_index());
    x_copy.set_scale(sumx.scale());
    ckks->evaluator.sub(x_copy, sumx, z[i]);
    ckks->evaluator.square(z[i], y[i]);
    ckks->evaluator.relinearize_inplace(y[i], *ckks->relin_keys);
    ckks->evaluator.rescale_to_next_inplace(y[i]);
  }
  for (int i=0; i<3; i++) {
    tmp[i] = quick_sum(y[i], ckks, 128);
  }
  PhantomCiphertext sumy, rot_sumy;
  ckks->evaluator.add_many(tmp, sumy);
  ckks->evaluator.rotate_vector(sumy, slot_count / 2, *(ckks->galois_keys), rot_sumy);
  ckks->evaluator.add_inplace(sumy, rot_sumy);

  PhantomPlaintext delta;
  ckks->encoder.encode(1.0 / 768, sumy.params_id(), sumy.scale(), delta);
  PhantomCiphertext inv_sqrt;
  ckks->evaluator.multiply_plain_inplace(sumy, delta);
  ckks->evaluator.rescale_to_next_inplace(sumy);

  inv_sqrt = ckks->invert_sqrt(sumy, 4, 2);

  for (int i=0; i<3; i++) {
    PhantomCiphertext a = z[i];
    ckks->evaluator.mod_switch_to_inplace(a, inv_sqrt.params_id());
    ckks->evaluator.multiply(inv_sqrt, a, res[i]);
    ckks->evaluator.relinearize_inplace(res[i], *ckks->relin_keys);
    ckks->evaluator.rescale_to_next_inplace(res[i]);
  }
}