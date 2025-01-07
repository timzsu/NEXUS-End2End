find_package(Catch2 REQUIRED)

macro(add_nn_test name)
  add_executable(${name}_test ${name}_test.cu)
  target_link_libraries(${name}_test
    PRIVATE
    NEXUS_NN
    Catch2::Catch2WithMain
  )
  target_precompile_headers(${name}_test PUBLIC ${PROJECT_SOURCE_DIR}/include/precompiled/catch2_includes.h)
  set_property(TARGET ${name}_test PROPERTY CUDA_STANDARD 20)
  add_test(NAME ${name}_test COMMAND ${name}_test)
endmacro()

macro(add_bert_test name)
  add_executable(${name}_test ${name}_test.cu)
  target_link_libraries(${name}_test
    PRIVATE
    BERT_MODULES
    Catch2::Catch2WithMain
  )
  target_precompile_headers(${name}_test PUBLIC ${PROJECT_SOURCE_DIR}/include/precompiled/catch2_includes.h)
  set_property(TARGET ${name}_test PROPERTY CUDA_STANDARD 20)
  add_test(NAME ${name}_test COMMAND ${name}_test)
endmacro()

# Enable testing
enable_testing()

add_nn_test(matmul)
add_nn_test(nonlinear)
add_bert_test(bert)