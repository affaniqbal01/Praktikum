cmake_minimum_required(VERSION 3.10)

# Project name
project(WarpAffine LANGUAGES CXX CUDA)

# Find OpenCV with CUDA support
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Set CUDA architecture (modify according to your GPU)
set(CUDA_ARCH_BIN "8.9" CACHE STRING "CUDA architecture bin")

# Include OpenCV and CUDA directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable
add_executable(WarpAffine src/main.cpp)

# Enable separable compilation for CUDA (optional, depending on your needs)
set(CUDA_SEPARABLE_COMPILATION ON)

# Link OpenCV and CUDA libraries
target_link_libraries(WarpAffine ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})
