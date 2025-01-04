#include "nn/matrix_mul.cuh"
#include "nn/gelu.cuh"
#include "nn/row_pack.h"

#include "ckks_evaluator.cuh"

namespace nexus {

class BertMLP {
private:
    MMEvaluator mm_evaluator;
    GELUEvaluator gelu_evaluator;
    std::shared_ptr<CKKSEvaluator> ckks;

    static constexpr int expansion_factor = 4;
    static constexpr int hidden_dim = 768;

    std::array<FlatVecMat, expansion_factor * hidden_dim> W_up_packed, W_down_packed;
    std::array<FlatVecArray, expansion_factor * hidden_dim> B_up_packed;
    FlatVecArray B_down_packed;

public:
    BertMLP(std::shared_ptr<CKKSEvaluator> ckks) : mm_evaluator(ckks), gelu_evaluator(ckks), ckks(ckks) {}

    void load_weights(Matrix W_up, Matrix W_down, Vector b_up, Vector b_down);

    void random_init();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
};

}