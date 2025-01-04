#include "bert/bert_attention.cuh"
#include "bert/bert_mlp.cuh"
#include "nn/layer_norm.cuh"

#include "ckks_evaluator.cuh"


namespace nexus {

class BertLayer {
private:
    BertAttention self_attention;
    BertMLP mlp;
    LNEvaluator ln_evaluator;
    std::shared_ptr<CKKSEvaluator> ckks;

public:
    BertLayer(std::shared_ptr<CKKSEvaluator> ckks): self_attention(ckks), mlp(ckks), ln_evaluator(ckks), ckks(ckks) {}
    
    void random_init();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
};

}