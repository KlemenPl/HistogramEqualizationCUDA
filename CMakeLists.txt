cmake_minimum_required(VERSION 3.24)
project(HistogramEqualization)
enable_language("CUDA")

set(CMAKE_CUDA_STANDARD 11)

add_executable(HistogramEqualization main.cu)
target_link_libraries(HistogramEqualization m)

