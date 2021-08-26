cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")
set(Torch_DIR "/home/wangshuoyuan/.local/lib/python3.6/site-packages/torch/lib/")
project(adder_cuda LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)
    
set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")


find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
 

#include_directories(/usr/bin/python3)
include_directories(../include)
 
#set(SRCS adder_cuda.cpp adder_cuda_kernel.cu)
#add_library(adder_cuda SHARED ${SRCS})

cuda_add_executable(adder_cuda adder_cuda.cpp adder_cuda_kernel.cu)
 
target_link_libraries(adder_cuda "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")