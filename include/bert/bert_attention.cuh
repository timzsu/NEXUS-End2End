#include "nn/matrix_mul.cuh"
#include "nn/softmax.cuh"
#include "nn/row_pack.h"

#include "ckks_evaluator.cuh"

namespace nexus {

class BertAttention {
private:
    MMEvaluator mm_evaluator;
    SoftmaxEvaluator softmax_evaluator;
    std::shared_ptr<CKKSEvaluator> ckks;

    static constexpr int num_heads = 12;

    std::array<FlatVecArray, num_heads/2> Wq_packed, Wk_packed, Wv_packed;
    std::array<FlatVec, num_heads/2> Bq_packed, Bk_packed, Bv_packed;
    FlatVecMat Wo_packed;
    FlatVecArray Bo_packed;

public:
    BertAttention(std::shared_ptr<CKKSEvaluator> ckks) : mm_evaluator(ckks), softmax_evaluator(ckks), ckks(ckks) {}

    void load_weights(Matrix Wq, Matrix Wk, Matrix Wv, Matrix Wo, Vector bq, Vector bk, Vector bv, Vector bo);

    void random_init();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
};

}