#include "bert/bert_mlp.cuh"
#include "nn/ckks_wrapper.cuh"

namespace nexus {

void BertMLP::pack_weights() {
    assert_shape(W_up, hidden_dim, hidden_dim*expansion_factor);
    assert_shape(W_down, hidden_dim*expansion_factor, hidden_dim);
    assert_shape(b_up, hidden_dim*expansion_factor);
    assert_shape(b_down, hidden_dim);

    for (int i=0; i<expansion_factor; i++) {
        W_up_packed[i] = row_pack_768x768(W_up.slice(1, i*hidden_dim, (i+1)*hidden_dim));
        B_up_packed[i] = row_pack_768x1(b_up.slice(0, i*hidden_dim, (i+1)*hidden_dim));
        W_down_packed[i] = row_pack_768x768(W_down.slice(1, i*hidden_dim, (i+1)*hidden_dim));
    }
    B_down_packed = row_pack_768x1(b_down);
}

std::vector<PhantomCiphertext> BertMLP::forward(vector<PhantomCiphertext>& x) {
    std::array<std::vector<PhantomCiphertext>, expansion_factor> x_mid, x_post_gelu;

    // Up projection
    for (int i=0; i<expansion_factor; i++) {
        mm_evaluator.matrix_mul_ct128x768_pt768x768(x, W_up_packed[i], x_mid[i]);
        for (int j=0; j<3; j++) {
            auto b_up_pt = CKKSEncode(B_up_packed[i][j], ckks, &x_mid[i][j]);
            ckks->evaluator.add_plain_inplace(x_mid[i][j], b_up_pt);
        }
    }
    for (int i=0; i<expansion_factor; i++) {
        for (int j=0; j<3; j++) {
            gelu_evaluator.gelu(x_mid[i][j], x_post_gelu[i][j]);
        }
    }
    
    // Down projection
    std::vector<PhantomCiphertext> out, tmp;
    for (int i=0; i<expansion_factor; i++) {
        mm_evaluator.matrix_mul_ct128x768_pt768x768(x_mid[i], W_down_packed[i], tmp);
        if (i == 0) {
            out = tmp;
        } else {
            for (int ct_idx=0; ct_idx<3; ct_idx++) {
                ckks->evaluator.add_inplace(out[ct_idx], tmp[ct_idx]);
            }
        }
    }
    for (int ct_idx=0; ct_idx<3; ct_idx++) {
        auto b_down_pt = CKKSEncode(B_down_packed[ct_idx], ckks, &out[ct_idx]);
        ckks->evaluator.add_plain_inplace(out[ct_idx], b_down_pt);
    }

    // Residual Link
    for (int ct_idx=0; ct_idx<3; ct_idx++) {
        ckks->evaluator.add_inplace(out[ct_idx], x[ct_idx]);
    }

    return out;
}

}