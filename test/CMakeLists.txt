macro(add_test_executable name)
  add_executable(${name}_test ${name}_test.cu)
  target_link_libraries(${name}_test
    PRIVATE
    PhantomBootstrapping::PhantomBootstrapping
    NEXUS_NN
  )
  add_test(NAME ${name}_test COMMAND ${name}_test)
endmacro()

add_test_executable(matmul)