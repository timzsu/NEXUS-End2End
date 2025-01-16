#pragma once

#include "ckks_evaluator.cuh"
#include "bootstrapping/Bootstrapper.cuh"
#include "nn/constant.cuh"

namespace nexus {

template<bool return_bootstrap = false>
auto setup() {
    EncryptionParameters parms(scheme_type::ckks);

    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);
    for (int i = 0; i < L; i++) {
        coeff_bit_vec.push_back(logp);
    }
    for (int i = 0; i < boot_level; i++) {
        coeff_bit_vec.push_back(logq);
    }
    coeff_bit_vec.push_back(log_special_prime);

    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(N, coeff_bit_vec));

    auto context = make_shared<PhantomContext>(parms);
    auto secret_key = make_shared<PhantomSecretKey>(*context);
    auto public_key = make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
    auto relin_keys = make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = make_shared<PhantomGaloisKey>(secret_key->create_galois_keys(*context));

    auto encoder = make_shared<PhantomCKKSEncoder>(*context);

    auto ckks_evaluator = make_shared<CKKSEvaluator>(context, public_key, secret_key, encoder, relin_keys, galois_keys, scale);

    if constexpr (return_bootstrap) {
        long boundary_K = 25;
        long deg = 59;
        long scale_factor = 2;
        long inverse_deg = 1;
        long loge = 10;
        auto bootstrapper = std::make_shared<Bootstrapper>(
            loge,
            logN,
            logN - 1,
            total_level,
            scale,
            boundary_K,
            deg,
            scale_factor,
            inverse_deg,
            ckks_evaluator);

        // Initialize the bootstrapper
        cout << "[Bootstrap Setup (1/4)] Generating Optimal Minimax Polynomials..." << endl;
        bootstrapper->prepare_mod_polynomial();

        cout << "[Bootstrap Setup (2/4)] Adding Bootstrapping Keys..." << endl;
        vector<int> gal_steps_vector;

        gal_steps_vector.push_back(0);
        for (int i = 0; i < logN - 1; i++) {
            gal_steps_vector.push_back((1 << i));
        }
        bootstrapper->addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

        std::cout << "[Bootstrap Setup (3/4)] Generating Galois keys from steps vector." << endl;
        // for (int i = 0; i < logN - 1; i++) {
        //     gal_steps_vector.push_back(-(1 << i));
        // }
        // ckks_evaluator->decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator->galois_keys));
        bootstrapper->slot_vec.push_back(logN - 1);

        cout << "[Bootstrap Setup (4/4)] Generating Linear Transformation Coefficients..." << endl;
        bootstrapper->generate_LT_coefficient_3();

        return std::make_pair(ckks_evaluator, bootstrapper);
    } else {
        return ckks_evaluator;
    }
}

}  // namespace nexus
