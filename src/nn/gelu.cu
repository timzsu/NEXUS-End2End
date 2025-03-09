#include "nn/gelu.cuh"
#include "nn/nexus_utility.cuh"

using namespace nexus;
using namespace phantom::arith;

void GELUEvaluator::gelu(PhantomCiphertext &x, PhantomCiphertext &res) {
  PhantomCiphertext b0, b1, b2;
  vector<double> dest;

  PhantomPlaintext p0 = CKKSEncode(ckks->init_vec_with_value(-3.5), ckks, &x);
  PhantomPlaintext p1 = CKKSEncode(ckks->init_vec_with_value(3.5), ckks, &x);
  PhantomPlaintext delta = CKKSEncode(ckks->init_vec_with_value(1.0 / 8.5), ckks, &x, true);

  ckks->evaluator.sub_plain(x, p0, b0);
  ckks->evaluator.multiply_plain_inplace(b0, delta);
  ckks->evaluator.rescale_to_next_inplace(b0);

  ckks->evaluator.sub_plain(x, p1, b1);
  ckks->evaluator.multiply_plain_inplace(b1, delta);
  ckks->evaluator.rescale_to_next_inplace(b1);

  bootstrap(b0, bootstrapper, true);
  bootstrap(b1, bootstrapper, true);
  b0 = ckks->sgn_eval(b0, d_g, d_f); // Consumes 11 levels
  b1 = ckks->sgn_eval(b1, d_g, d_f); // Consumes 11 levels
  bootstrap(b0, bootstrapper);
  bootstrap(b1, bootstrapper);

  PhantomPlaintext zero_point_five = CKKSEncode(ckks->init_vec_with_value(0.5), ckks, &b1);
  PhantomCiphertext a0, a1, a2;

  ckks->evaluator.sub(b0, b1, a1);                     // a1 = b0 - b1
  ckks->evaluator.add_plain(b1, zero_point_five, a2);  // a2 = b1 + 0.5

  PhantomCiphertext x_2;
  ckks->evaluator.square(x, x_2);
  ckks->evaluator.relinearize_inplace(x_2, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(x_2);

  PhantomCiphertext x_4;
  ckks->evaluator.square(x_2, x_4);
  ckks->evaluator.relinearize_inplace(x_4, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(x_4);

  PhantomCiphertext x_6;
  ckks->evaluator.mod_switch_to_inplace(x_2, x_4.params_id());
  ckks->evaluator.multiply(x_2, x_4, x_6);
  ckks->evaluator.relinearize_inplace(x_6, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(x_6);

  PhantomCiphertext x_8;
  ckks->evaluator.square(x_4, x_8);
  ckks->evaluator.relinearize_inplace(x_8, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(x_8);

  PhantomCiphertext x_10;
  ckks->evaluator.mod_switch_to_inplace(x_2, x_8.params_id());
  ckks->evaluator.multiply(x_2, x_8, x_10);
  ckks->evaluator.relinearize_inplace(x_10, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(x_10);

  PhantomCiphertext x_12;
  ckks->evaluator.square(x_6, x_12);
  ckks->evaluator.relinearize_inplace(x_12, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(x_12);

  double A[] = {2.25775755e-04, 0.5, 3.96880960e-01, -6.37042698e-02, 8.38841647e-03, -7.17830961e-04, 3.49617829e-05, -7.26059653e-07};
  vector<PhantomPlaintext> coeff_A(8);
  for (size_t i = 0; i < coeff_A.size(); i++) {
    coeff_A[i] = CKKSEncode(A[i], ckks);
  }
  vector<PhantomCiphertext> cts(8);
  cts[1] = x;
  cts[2] = x_2;
  cts[3] = x_4;
  cts[4] = x_6;
  cts[5] = x_8;
  cts[6] = x_10;
  cts[7] = x_12;

  // Ax = A[0]+A[1]x+A[2]x^2+A[3]x^4+A[4]x^6+A[5]x^8+A[6]x^10+A[7]x^12
  for (size_t i = 1; i < coeff_A.size(); i++) {
    ckks->evaluator.mod_switch_to_inplace(coeff_A[i], cts[i].params_id());
    ckks->evaluator.multiply_plain_inplace(cts[i], coeff_A[i]);
    ckks->evaluator.rescale_to_next_inplace(cts[i]);
    cts[i].scale() = ckks->scale;
  }

  PhantomCiphertext Ax = cts[cts.size() - 1];

  for (size_t i = 1; i < coeff_A.size() - 1; i++) {
    ckks->evaluator.mod_switch_to_inplace(cts[i], Ax.params_id());
    ckks->evaluator.add_inplace(Ax, cts[i]);
  }

  ckks->evaluator.mod_switch_to_inplace(coeff_A[0], Ax.params_id());
  ckks->evaluator.add_plain_inplace(Ax, coeff_A[0]);

  PhantomCiphertext s1, s2;
  // // cout << Ax.scale() << " " << Bx.scale() << " " << a1.scale() << endl;
  mod_switch_to_same(Ax, a1, ckks);
  ckks->evaluator.multiply(Ax, a1, s1);
  ckks->evaluator.relinearize_inplace(s1, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(s1);

  mod_switch_to_same(x, a2, ckks);
  ckks->evaluator.multiply(x, a2, s2);
  ckks->evaluator.relinearize_inplace(s2, *ckks->relin_keys);
  ckks->evaluator.rescale_to_next_inplace(s2);

  s1.scale() = ckks->scale;
  s2.scale() = ckks->scale;
  mod_switch_to_same(s1, s2, ckks);
  ckks->evaluator.add(s1, s2, res);

  // cout << "Moduli left after GELU: " << res.coeff_modulus_size() << endl;
}
