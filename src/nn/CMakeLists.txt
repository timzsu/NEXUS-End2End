add_library(NEXUS_NN SHARED
    argmax.cu
    gelu.cu
    layer_norm.cu 
    matrix_mul.cu
    row_pack.cpp
    softmax.cu    
)
target_include_directories(NEXUS_NN
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(NEXUS_NN
  PUBLIC
  PhantomBootstrapping::PhantomBootstrapping
  Eigen3::Eigen
)