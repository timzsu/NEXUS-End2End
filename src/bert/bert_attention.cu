#include "bert/bert_attention.cuh"
#include "nn/ckks_wrapper.cuh"

namespace nexus {

void BertAttention::pack_weights() {
    assert_shape(Wq, 768, 768);
    assert_shape(Wk, 768, 768);
    assert_shape(Wv, 768, 768);
    assert_shape(Wo, 768, 768);
    assert_shape(bq, 768);
    assert_shape(bk, 768);
    assert_shape(bv, 768);
    assert_shape(bo, 768);

    std::vector<std::pair<int, int>> orders{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}};
    for (int i=0; i<orders.size(); i++) {
        auto& [lhs, rhs] = orders[i];
        Wq_packed[i] = row_pack_768x64x2(Wq.slice(1, lhs*64, (lhs+1)*64), Wq.slice(1, rhs*64, (rhs+1)*64));
        Wk_packed[i] = row_pack_768x64x2(Wk.slice(1, lhs*64, (lhs+1)*64), Wk.slice(1, rhs*64, (rhs+1)*64));
        Wv_packed[i] = row_pack_768x128(torch::concat({Wv.slice(1, lhs*64, (lhs+1)*64), Wv.slice(1, rhs*64, (rhs+1)*64)}, 1));
        Bq_packed[i] = row_pack_64x1x2(bq.slice(0, lhs*64, (lhs+1)*64), bq.slice(0, rhs*64, (rhs+1)*64));
        Bk_packed[i] = row_pack_64x1x2(bk.slice(0, lhs*64, (lhs+1)*64), bk.slice(0, rhs*64, (rhs+1)*64));
        Bv_packed[i] = row_pack_128x1(torch::concat({bv.slice(0, lhs*64, (lhs+1)*64), bv.slice(0, rhs*64, (rhs+1)*64)}, 1));
    }
    Wo_packed = row_pack_768x768(Wo);
    Bo_packed = row_pack_768x1(bo);
}

std::vector<PhantomCiphertext> BertAttention::forward(vector<PhantomCiphertext>& x) {
    // Implement the forward pass for self-attention here
    std::array<PhantomCiphertext, num_heads/2> Q, K, V, QK, So, QKV;
    for (int i=0; i<num_heads/2; i++) {
        mm_evaluator.matrix_mul_ct128x768_pt768x64x2(x, Wq_packed[i], Q[i]);
        mm_evaluator.matrix_mul_ct128x768_pt768x64x2(x, Wk_packed[i], K[i]);
        mm_evaluator.matrix_mul_ct128x768_pt768x128(x, Wv_packed[i], V[i]);
        auto bq_pt = CKKSEncode(Bq_packed[i], ckks, &Q[i]);
        ckks->evaluator.add_plain_inplace(Q[i], bq_pt);
        auto bk_pt = CKKSEncode(Bk_packed[i], ckks, &K[i]);
        ckks->evaluator.add_plain_inplace(K[i], bk_pt);
        auto bv_pt = CKKSEncode(Bv_packed[i], ckks, &V[i]);
        ckks->evaluator.add_plain_inplace(V[i], bv_pt);

        mm_evaluator.matrix_mul_ct128x64_ct128x64_transpose(Q[i], K[i], QK[i]);
        softmax_evaluator.softmax(QK[i], So[i], 128);
        mm_evaluator.matrix_mul_ct128x128_ct128x128(So[i], V[i], QKV[i]);
    }
    std::vector<PhantomCiphertext> attention_value(num_heads/4), attn_output;
    for (int i=0; i<num_heads/4; i++) {
        ckks->evaluator.rotate_vector_inplace(QKV[2*i+1], MMEvaluator::slot_count/2, *(ckks->galois_keys));
        ckks->evaluator.add(QKV[2*i], QKV[2*i+1], attention_value[i]);
    }
    mm_evaluator.matrix_mul_ct128x768_pt768x768(attention_value, Wo_packed, attn_output);
    for (int i=0; i<3; i++) {
        auto bo_pt = CKKSEncode(Bo_packed[i], ckks, &attn_output[i]);
        ckks->evaluator.add_plain_inplace(attn_output[i], bo_pt);
    }
    return attn_output;
}

}