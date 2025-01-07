#include "nn/matrix_mul.cuh"
#include "nn/softmax.cuh"
#include "nn/row_pack.h"

#include "ckks_evaluator.cuh"

namespace nexus {

class BertAttention: torch::nn::Module {
private:
    MMEvaluator mm_evaluator;
    SoftmaxEvaluator softmax_evaluator;
    std::shared_ptr<CKKSEvaluator> ckks;

    static constexpr int num_heads = 12;

    std::array<FlatVecArray, num_heads/2> Wq_packed, Wk_packed, Wv_packed;
    std::array<FlatVec, num_heads/2> Bq_packed, Bk_packed, Bv_packed;
    FlatVecMat Wo_packed;
    FlatVecArray Bo_packed;

    torch::Tensor Wq, Wk, Wv, Wo;
    torch::Tensor bq, bk, bv, bo;

public:
    BertAttention(std::shared_ptr<CKKSEvaluator> ckks) : mm_evaluator(ckks), softmax_evaluator(ckks), ckks(ckks) {
        Wq = torch::randn({768, 768}, torch::kDouble);
        Wk = torch::randn({768, 768}, torch::kDouble);
        Wv = torch::randn({768, 768}, torch::kDouble);
        Wo = torch::randn({768, 768}, torch::kDouble);
        bq = torch::randn({768}, torch::kDouble);
        bk = torch::randn({768}, torch::kDouble);
        bv = torch::randn({768}, torch::kDouble);
        bo = torch::randn({768}, torch::kDouble);

        register_parameter("weight_q", Wq, false);
        register_parameter("weight_k", Wk, false);
        register_parameter("weight_v", Wv, false);
        register_parameter("weight_o", Wo, false);
        register_parameter("bias_q", bq, false);
        register_parameter("bias_k", bk, false);
        register_parameter("bias_v", bv, false);
        register_parameter("bias_o", bo, false);
    }

    void pack_weights();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
};

}