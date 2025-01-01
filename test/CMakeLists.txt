find_package(Catch2 REQUIRED)

macro(add_test_executable name)
  add_executable(${name}_test ${name}_test.cu)
  target_link_libraries(${name}_test
    PRIVATE
    PhantomBootstrapping::PhantomBootstrapping
    NEXUS_NN
    Eigen3::Eigen
    Catch2::Catch2WithMain
  )
  # target_compile_options(${name}_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
  set_property(TARGET ${name}_test PROPERTY CUDA_STANDARD 20)
  add_test(NAME ${name}_test COMMAND ${name}_test)
endmacro()

# Enable testing
enable_testing()

add_test_executable(matmul)