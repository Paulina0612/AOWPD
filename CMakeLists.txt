cmake_minimum_required(VERSION 3.12)

# Set the project name and version
project(RadixSort VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SRC
    main.cpp
    cpu_radix_sort.cpp
    cpu_radix_parallel.cpp
)

# Add the executable target
add_executable(RadixSort ${SRC})

set(BASE 10 CACHE STRING "Base to use while building")

target_compile_definitions(RadixSort
    PRIVATE
    BASE=${BASE}
)
