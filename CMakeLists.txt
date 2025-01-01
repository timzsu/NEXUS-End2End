cmake_minimum_required(VERSION 3.26)

project(NEXUS-end2end LANGUAGES CXX CUDA)

find_package(PhantomBootstrapping 1.0 COMPONENTS PhantomBootstrapping REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CUDA_FLAGS "-G -maxrregcount 64")

add_subdirectory(src)
add_subdirectory(test)