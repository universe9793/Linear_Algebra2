cmake_minimum_required(VERSION 3.22)
project(itebd)

set(Torch_DIR "\\home\\libtorch\\share\\cmake\\Torch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_compile_options(-O3)

set(CMAKE_CXX_STANDARD 17)
add_executable(itebd main.cpp)
target_link_libraries(itebd "${TORCH_LIBRARIES}")
set_property(TARGET itebd PROPERTY CXX_STANDARD 17)

