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

    Timer bootstrap1_timer, layer_norm1_timer, bootstrap2_timer, bootstrap3_timer, layer_norm2_timer, bootstrap4_timer;

public:
    BertLayer(std::shared_ptr<CKKSEvaluator> ckks, std::shared_ptr<Bootstrapper> bootstrapper): self_attention(ckks), mlp(ckks), ln_evaluator(ckks), ckks(ckks), bootstrapper(bootstrapper) {}

    void pack_weights();

    void bootstrap(PhantomCiphertext &x);
    void bootstrap(std::vector<PhantomCiphertext> &x);

    std::vector<PhantomCiphertext> forward(vector<PhantomCiphertext>& x);
    torch::Tensor forward(torch::Tensor x);

    void print_time() {
        self_attention.print_time();
        cout << "bootstrap1 takes " << bootstrap1_timer.duration() << "ms" << endl;
        cout << "ln1 takes " << layer_norm1_timer.duration() << "ms" << endl;
        cout << "bootstrap2 takes " << bootstrap2_timer.duration() << "ms" << endl;
        mlp.print_time();
        cout << "bootstrap3 takes " << bootstrap3_timer.duration() << "ms" << endl;
        cout << "ln2 takes " << layer_norm2_timer.duration() << "ms" << endl;
        cout << "bootstrap4 takes " << bootstrap4_timer.duration() << "ms" << endl;
    }
};

}