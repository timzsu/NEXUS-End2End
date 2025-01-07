#include "nn/matrix_mul.cuh"
#include "nn/gelu.cuh"
#include "nn/row_pack.h"

#include "ckks_evaluator.cuh"

namespace nexus {

class BertMLP: torch::nn::Module {
private:
    MMEvaluator mm_evaluator;
    GELUEvaluator gelu_evaluator;
    std::shared_ptr<CKKSEvaluator> ckks;

    static constexpr int expansion_factor = 4;
    static constexpr int hidden_dim = 768;

    torch::Tensor W_up, W_down, b_up, b_down;

    std::array<FlatVecMat, expansion_factor * hidden_dim> W_up_packed, W_down_packed;
    std::array<FlatVecArray, expansion_factor * hidden_dim> B_up_packed;
    FlatVecArray B_down_packed;

public:
    BertMLP(std::shared_ptr<CKKSEvaluator> ckks) : mm_evaluator(ckks), gelu_evaluator(ckks), ckks(ckks) {
        W_up = torch::randn({hidden_dim, hidden_dim*expansion_factor}, torch::kDouble);
        W_down = torch::randn({hidden_dim*expansion_factor, hidden_dim}, torch::kDouble);
        b_up = torch::randn({hidden_dim*expansion_factor}, torch::kDouble);
        b_down = torch::randn({hidden_dim}, torch::kDouble);

        register_parameter("weight_up", W_up, false);
        register_parameter("weight_down", W_down, false);
        register_parameter("bias_up", b_up, false);
        register_parameter("bias_down", b_down, false);
    }

    void pack_weights();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
};

}