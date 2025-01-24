#include "nn/matrix_mul.cuh"
#include "nn/softmax.cuh"
#include "nn/row_pack.h"
#include "utils.cuh"

#include "ckks_evaluator.cuh"

namespace nexus {

class BertAttention: torch::nn::Module {
private:
    MMEvaluator mm_evaluator;
    SoftmaxEvaluator softmax_evaluator;
    std::shared_ptr<CKKSEvaluator> ckks;

    static constexpr int num_heads = 12;
    static constexpr int embed_dim = 768;
    static constexpr int head_dim = embed_dim / num_heads;

    torch::nn::Linear q_proj, k_proj, v_proj, o_proj;

    std::array<FlatVecArray, num_heads/2> Wq_packed, Wk_packed, Wv_packed;
    std::array<FlatVec, num_heads/2> Bq_packed, Bk_packed, Bv_packed;
    FlatVecMat Wo_packed;
    FlatVecArray Bo_packed;

    long qkv_proj_time=0, qk_time=0, softmax_time=0, qkv_time=0;
    Timer o_proj_timer;

public:
    BertAttention(std::shared_ptr<CKKSEvaluator> ckks) : mm_evaluator(ckks), softmax_evaluator(ckks), ckks(ckks), 
    q_proj(torch::nn::LinearOptions(embed_dim, embed_dim)), 
    k_proj(torch::nn::LinearOptions(embed_dim, embed_dim)), 
    v_proj(torch::nn::LinearOptions(embed_dim, embed_dim)), 
    o_proj(torch::nn::LinearOptions(embed_dim, embed_dim)) {
        torch::nn::init::uniform_(o_proj->weight, -5, 5);
    }

    void pack_weights();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
    torch::Tensor forward(torch::Tensor x);

    void print_time() {
        cout << "qkv projection takes " << qkv_proj_time << "ms" << endl;
        cout << "q * k^T takes " << qk_time << "ms" << endl;
        cout << "softmax takes " << softmax_time << "ms" << endl;
        cout << "qk^T * v takes " << qkv_time << "ms" << endl;
        cout << "out projection takes " << o_proj_timer.duration() << "ms" << endl;
    }
};

}