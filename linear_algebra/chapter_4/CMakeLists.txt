cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ch4_1)

add_compile_options(-O3 -mavx -mfma)
add_executable(ch4_1 ch4_1.cpp)
set_property(TARGET ch4_1 PROPERTY CXX_STANDARD 17)

