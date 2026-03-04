cmake_minimum_required(VERSION 3.25)
project(Yolo26 LANGUAGES CXX CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)

include(FindPkgConfig)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio highgui imgproc)

###################################################################

add_library(yolo26 STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bbox.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pose.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/segm.cu
)

target_include_directories(yolo26
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_BINARY_DIR}
)

target_link_libraries(yolo26
    PUBLIC
        corekit-core
        corekit-cuda
)

add_library(yolo26::yolo26 ALIAS yolo26)

###################################################################

add_executable(yolo26_app
    ${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp
)

target_include_directories(yolo26_app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_BINARY_DIR}
)

target_link_libraries(yolo26_app
    PRIVATE
        yolo26
)

# NVCC struggles with C++20 ranges used by nlohmann::json; disable ranges in CUDA builds.
target_compile_definitions(yolo26 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:JSON_HAS_RANGES=0>
)