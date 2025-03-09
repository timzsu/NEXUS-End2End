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

    Timer layer_norm1_timer, layer_norm2_timer;

public:
    BertLayer(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper): self_attention(ckks, bootstrapper), mlp(ckks, bootstrapper), ln_evaluator(ckks, bootstrapper), ckks(ckks), bootstrapper(bootstrapper) {}

    void pack_weights();

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x, FlatVec attention_mask);
    torch::Tensor forward(torch::Tensor x, torch::Tensor attention_mask);

    void print_time() {
        self_attention.print_time();
        cout << "ln1 takes " << layer_norm1_timer.duration() << "ms" << endl;
        mlp.print_time();
        cout << "ln2 takes " << layer_norm2_timer.duration() << "ms" << endl;
    }
};

}