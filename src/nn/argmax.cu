#include "nn/argmax.cuh"

void ArgmaxEvaluator::argmax(PhantomCiphertext &x, PhantomCiphertext &x_copy, int len) {
  PhantomCiphertext tmp, b, sign, a_plus_b, a_minus_b, a_minus_b_sgn;
  PhantomPlaintext one, half;

  int log_step = log2(len);
  x_copy = x;

  // Transform x = [a_0, ..., a_n, 0, ..., 0] to [a_0, ..., a_n, a_0, ..., a_n, 0, ..., 0]
  ckks->evaluator.rotate_vector(x, -len, *(ckks->galois_keys), tmp);
  ckks->evaluator.add_inplace(x, tmp);

  // QuickMax
  for (int i = 0; i < log_step; ++i) {
    ckks->evaluator.rotate_vector(x, pow(2, i), *ckks->galois_keys, b);

    ckks->evaluator.add(x, b, a_plus_b);
    ckks->evaluator.sub(x, b, a_minus_b);
    sign = ckks->sgn_eval(a_minus_b, 2, 2);

    // (a - b) * sgn(a - b) / 2
    ckks->evaluator.mod_switch_to_inplace(a_minus_b, sign.params_id());
    ckks->evaluator.multiply(a_minus_b, sign, a_minus_b_sgn);
    ckks->evaluator.relinearize_inplace(a_minus_b_sgn, *ckks->relin_keys);
    ckks->evaluator.rescale_to_next_inplace(a_minus_b_sgn);

    // (a + b) / 2
    ckks->encoder.encode(0.5, a_plus_b.params_id(), a_plus_b.scale(), half);
    ckks->evaluator.multiply_plain_inplace(a_plus_b, half);
    ckks->evaluator.rescale_to_next_inplace(a_plus_b);

    // a = max(a, b)
    a_plus_b.scale() = a_minus_b_sgn.scale();
    ckks->evaluator.mod_switch_to_inplace(a_plus_b, a_minus_b_sgn.params_id());
    ckks->evaluator.add(a_plus_b, a_minus_b_sgn, x);

    bootstrap(x);
  }

  x_copy.scale() = x.scale();
  ckks->evaluator.mod_switch_to_inplace(x_copy, x.chain_index());
  ckks->evaluator.sub_inplace(x_copy, x);

  x_copy = ckks->sgn_eval(x_copy, 2, 2, 1.0);

  ckks->encoder.encode(1.0, x_copy.params_id(), x_copy.scale(), one);
  ckks->evaluator.add_plain_inplace(x_copy, one);
}

void ArgmaxEvaluator::bootstrap(PhantomCiphertext &x) {
  while (x.coeff_modulus_size() > 1) {
    ckks->evaluator.mod_switch_to_next_inplace(x);
  }
  PhantomCiphertext rtn;
  bootstrapper->set_final_scale(x.scale());
  bootstrapper->bootstrap_3(rtn, x);
  x = rtn;
}
