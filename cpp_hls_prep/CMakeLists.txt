cmake_minimum_required(VERSION 3.20)
project(tcn_hls_prep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tcn_fixed_core
    src/int8_kernels.cpp
    src/fixed_tcn_host.cpp
    src/fixed_tcn_modules.cpp
    src/fixed_tcn.cpp
    src/fixed_tcn_top.cpp
)

target_include_directories(tcn_fixed_core PUBLIC include)

add_executable(tcn_hls_prep src/main.cpp)
target_link_libraries(tcn_hls_prep PRIVATE tcn_fixed_core)

add_executable(tcn_hls_testbench src/hls_testbench.cpp)
target_link_libraries(tcn_hls_testbench PRIVATE tcn_fixed_core)
