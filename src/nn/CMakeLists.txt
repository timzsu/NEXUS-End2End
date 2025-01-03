add_library(NEXUS_NN SHARED
    argmax.cu
    gelu.cu
    layer_norm.cu 
    matrix_mul.cu
    softmax.cu    
)
target_include_directories(NEXUS_NN
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/nn>
  $<INSTALL_INTERFACE:include/nn>
)
target_link_libraries(NEXUS_NN
  PUBLIC
  PhantomBootstrapping::PhantomBootstrapping
  Eigen3::Eigen
)