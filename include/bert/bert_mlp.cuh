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

    torch::nn::Linear up_proj, down_proj;
    torch::nn::GELU gelu;

    std::array<FlatVecMat, expansion_factor * hidden_dim> W_up_packed, W_down_packed;
    std::array<FlatVecArray, expansion_factor * hidden_dim> B_up_packed;
    FlatVecArray B_down_packed;

public:
    BertMLP(std::shared_ptr<CKKSEvaluator> ckks) : mm_evaluator(ckks), gelu_evaluator(ckks), ckks(ckks), 
        up_proj(torch::nn::LinearOptions(hidden_dim, hidden_dim*expansion_factor)), 
        down_proj(torch::nn::LinearOptions(hidden_dim*expansion_factor, hidden_dim)),
        gelu(torch::nn::GELUOptions()) {}

    void pack_weights();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
    torch::Tensor forward(torch::Tensor x) {
        auto out = up_proj->forward(x);
        out = gelu->forward(out);
        out = down_proj->forward(out);
        out += x;
        return out;
    }
};

}