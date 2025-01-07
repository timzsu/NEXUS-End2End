#include "bert/bert_attention.cuh"
#include "bert/bert_mlp.cuh"
#include "nn/layer_norm.cuh"

#include "ckks_evaluator.cuh"
#include "bootstrapping/Bootstrapper.cuh"


namespace nexus {

class BertLayer : torch::nn::Module {
private:
    BertAttention self_attention;
    BertMLP mlp;
    LNEvaluator ln_evaluator;
    std::shared_ptr<CKKSEvaluator> ckks;
    std::shared_ptr<Bootstrapper> bootstrapper;

public:
    BertLayer(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper): self_attention(ckks), mlp(ckks), ln_evaluator(ckks), ckks(ckks), bootstrapper(bootstrapper) {}

    void pack_weights();

    void bootstrap(PhantomCiphertext &x);

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
};

}