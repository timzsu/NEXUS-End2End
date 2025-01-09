add_executable(bert_inference bert_inference.cu)
target_link_libraries(bert_inference
  PRIVATE
  BERT_MODULES
)