add_library(BERT_MODULES SHARED
    bert_attention.cu
)
target_link_libraries(BERT_MODULES PUBLIC NEXUS_NN)