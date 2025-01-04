#include "bert/bert_attention.cuh"
#include "nn/ckks_wrapper.cuh"

namespace nexus {

void BertAttention::load_weights(Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, Vector bq, Vector bk, Vector bv, Vector bo) {
    assert_shape(Wq, 768, 768);
    assert_shape(Wk, 768, 768);
    assert_shape(Wv, 768, 768);
    assert_shape(Wo, 768, 768);
    assert_shape(bq, 768);
    assert_shape(bk, 768);
    assert_shape(bv, 768);
    assert_shape(bo, 768);

    for (int i=0; i<num_heads; i+=2) {
        Wq_packed[i/2] = row_pack_768x64x2(Wq.block(0, i*64, 768, 64), Wq.block(0, (i+1)*64, 768, 64));
        Wk_packed[i/2] = row_pack_768x64x2(Wk.block(0, i*64, 768, 64), Wk.block(0, (i+1)*64, 768, 64));
        Wv_packed[i/2] = row_pack_768x128(Wv.block(0, i*64, 768, 128));
        Bq_packed[i/2] = row_pack_64x1x2(bq.segment(i*64, 64), bq.segment((i+1)*64, 64));
        Bk_packed[i/2] = row_pack_64x1x2(bk.segment(i*64, 64), bk.segment((i+1)*64, 64));
        Bv_packed[i/2] = row_pack_128x1(bv.segment(i*64, 128));
    }
    Wo_packed = row_pack_768x768(Wo);
    Bo_packed = row_pack_768x1(bo);
}

void BertAttention::random_init() {
    load_weights(
        Matrix::Random(768, 768),
        Matrix::Random(768, 768),
        Matrix::Random(768, 768),
        Matrix::Random(768, 768),
        Vector::Random(768),
        Vector::Random(768),
        Vector::Random(768),
        Vector::Random(768)
    );
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