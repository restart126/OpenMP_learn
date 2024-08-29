cmake_minimum_required(VERSION 3.27)
project(openmp_learn)

set(CMAKE_CXX_STANDARD 20)

# 设置编译结果的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../build)

# 手动设置 OpenMP 链接选项
if(MSVC)
    set(OpenMP_FLAGS "/openmp")
    set(OpenMP_LIBS "")
else()
    set(OpenMP_FLAGS "-fopenmp")
    set(OpenMP_LIBS "gomp")
endif()

# 添加编译标志
add_compile_options(${OpenMP_FLAGS})

# 为 exercise2.cpp 创建一个可执行文件，并添加 OpenMP 支持
add_executable(exercise2_program src/exercise2.cpp
        src/lesson18.cpp)
target_link_libraries(exercise2_program ${OpenMP_LIBS})
