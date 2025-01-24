#include "bert/bert_attention.cuh"
#include "nn/nexus_utility.cuh"
#include "nn/row_pack.h"
#include "nn/constant.cuh"

namespace nexus {

void BertAttention::pack_weights() {

    torch::Tensor Wq = q_proj->weight.transpose(0, 1).to(torch::kDouble);
    torch::Tensor Wk = k_proj->weight.transpose(0, 1).to(torch::kDouble);
    torch::Tensor Wv = v_proj->weight.transpose(0, 1).to(torch::kDouble);
    torch::Tensor Wo = o_proj->weight.transpose(0, 1).to(torch::kDouble);
    torch::Tensor bq = q_proj->bias.to(torch::kDouble);
    torch::Tensor bk = k_proj->bias.to(torch::kDouble);
    torch::Tensor bv = v_proj->bias.to(torch::kDouble);
    torch::Tensor bo = o_proj->bias.to(torch::kDouble);

    assert_shape(Wq, {768, 768});
    assert_shape(Wk, {768, 768});
    assert_shape(Wv, {768, 768});
    assert_shape(Wo, {768, 768});
    assert_shape(bq, 768);
    assert_shape(bk, 768);
    assert_shape(bv, 768);
    assert_shape(bo, 768);

    std::vector<std::pair<int, int>> orders{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}};
    for (int i=0; i<num_heads/2; i++) {
        auto& [lhs, rhs] = orders[i];
        Wq_packed[i] = row_pack_768x64x2(Wq.slice(1, lhs*64, (lhs+1)*64), Wq.slice(1, rhs*64, (rhs+1)*64));
        Wk_packed[i] = row_pack_768x64x2(Wk.slice(1, lhs*64, (lhs+1)*64), Wk.slice(1, rhs*64, (rhs+1)*64));
        Wv_packed[i] = row_pack_768x128(torch::concat({Wv.slice(1, lhs*64, (lhs+1)*64), Wv.slice(1, rhs*64, (rhs+1)*64)}, 1));
        Bq_packed[i] = row_pack_64x1x2(bq.slice(0, lhs*64, (lhs+1)*64), bq.slice(0, rhs*64, (rhs+1)*64));
        Bk_packed[i] = row_pack_64x1x2(bk.slice(0, lhs*64, (lhs+1)*64), bk.slice(0, rhs*64, (rhs+1)*64));
        Bv_packed[i] = row_pack_128x1(torch::concat({bv.slice(0, lhs*64, (lhs+1)*64), bv.slice(0, rhs*64, (rhs+1)*64)}, 0));
    }
    Wo_packed = row_pack_768x768(Wo);
    Bo_packed = row_pack_768x1(bo);
}

std::vector<PhantomCiphertext> BertAttention::forward(vector<PhantomCiphertext>& x) {
    // Implement the forward pass for self-attention here
    Timer timer;
    std::array<PhantomCiphertext, num_heads/2> QKV;
    for (int i=0; i<num_heads/2; i++) {
        timer.start();
        PhantomCiphertext Q, K, V;
        mm_evaluator.matrix_mul_ct128x768_pt768x64x2(x, Wq_packed[i], Q);
        mm_evaluator.matrix_mul_ct128x768_pt768x64x2(x, Wk_packed[i], K);
        mm_evaluator.matrix_mul_ct128x768_pt768x128(x, Wv_packed[i], V);
        PhantomPlaintext bq_pt = CKKSEncode(Bq_packed[i], ckks, &Q);
        ckks->evaluator.add_plain_inplace(Q, bq_pt);
        PhantomPlaintext bk_pt = CKKSEncode(Bk_packed[i], ckks, &K);
        ckks->evaluator.add_plain_inplace(K, bk_pt);
        PhantomPlaintext bv_pt = CKKSEncode(Bv_packed[i], ckks, &V);
        ckks->evaluator.add_plain_inplace(V, bv_pt);
        torch::cuda::synchronize();
        timer.stop();
        qkv_proj_time += timer.duration();
        
        timer.start();
        PhantomCiphertext QK, So;
        mm_evaluator.matrix_mul_ct128x64_ct128x64_transpose(Q, K, QK);
        std::vector<double> ratio(slot_count, 1./std::sqrt(head_dim));
        ckks->evaluator.multiply_vector_inplace_reduced_error(QK, ratio);
        ckks->evaluator.rescale_to_next_inplace(QK);
        torch::cuda::synchronize();
        timer.stop();
        qk_time += timer.duration();
        timer.start();
        softmax_evaluator.softmax_128x128(QK, So);
        torch::cuda::synchronize();
        timer.stop();
        softmax_time += timer.duration();
        timer.start();
        mm_evaluator.matrix_mul_ct128x128_ct128x128(So, V, QKV[i]);
        torch::cuda::synchronize();
        timer.stop();
        qkv_time += timer.duration();
    }
    std::vector<PhantomCiphertext> attn_weight(num_heads/4), attn_output;
    for (int i=0; i<num_heads/4; i++) {
        ckks->evaluator.rotate_vector_inplace(QKV[2*i+1], slot_count/2, *(ckks->galois_keys));
        ckks->evaluator.add(QKV[2*i], QKV[2*i+1], attn_weight[i]);
    }
    
    torch::cuda::synchronize();
    o_proj_timer.start();
    mm_evaluator.matrix_mul_ct128x768_pt768x768(attn_weight, Wo_packed, attn_output);
    for (int i=0; i<3; i++) {
        auto bo_pt = CKKSEncode(Bo_packed[i], ckks, &attn_output[i]);
        ckks->evaluator.add_plain_inplace(attn_output[i], bo_pt);
    }
    torch::cuda::synchronize();
    o_proj_timer.stop();
    return attn_output;
}

// Reference: https://discuss.pytorch.org/t/which-multihead-attention-implementation-is-correct/198996/2
torch::Tensor BertAttention::forward(torch::Tensor x) {
    TORCH_CHECK(x.sizes().size() == 2 || x.sizes().size() == 3, "x should have 2 or 3 dimensions, but the input has", x.sizes().size(), "dimensions");
    bool no_batch_flag = (x.sizes().size() == 2);
    if (no_batch_flag) {
        x.unsqueeze_(0);
    }

    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int embed_size = x.size(2);
        
    auto query = q_proj->forward(x).view({batch_size, seq_len, num_heads, head_dim}).transpose(1,2);
    auto key = k_proj->forward(x).view({batch_size, seq_len, num_heads, head_dim}).transpose(1,2);
    auto value = v_proj->forward(x).view({batch_size, seq_len, num_heads, head_dim}).transpose(1,2);

    auto scores = torch::matmul(query, key.transpose(-2,-1))/ std::sqrt(head_dim);
    // if mask is not None:
    //     scores.masked_fill(mask==0, float("-inf"))
    auto attn_weight = torch::softmax(scores, -1);
    
    auto attention = torch::matmul(attn_weight, value);
    attention = attention.transpose(1,2).contiguous().view({batch_size, seq_len, embed_size});
    auto attn_output = o_proj->forward(attention);

    if (no_batch_flag) {
        attn_output.squeeze_(0);
    }
    return attn_output;
}

}