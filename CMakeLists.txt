cmake_minimum_required(VERSION 3.16)
project(CIFAR10_Inference_Engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for MinGW GCC
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -march=native)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Operators/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Test_Operators/include
)

# Source files
file(GLOB UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/utilities/src/*.cpp)
file(GLOB OP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Operators/src/*.cpp)

# Main Inference Application (Supports both --mode test and --mode model)
add_executable(cifar10_engine
    src/main.cpp
    Test_Operators/src/test_suite.cpp
    ${UTILS_SOURCES}
    ${OP_SOURCES}
)

# Standalone Unit Testing Executable
add_executable(run_unit_tests
    Test_Operators/src/test_main.cpp
    Test_Operators/src/test_suite.cpp
    ${UTILS_SOURCES}
    ${OP_SOURCES}
)