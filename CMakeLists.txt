cmake_minimum_required(VERSION 3.7)
project(semantic_project)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES draw.cpp)
add_executable(semantic_project ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(semantic_project ${OpenCV_LIBS} yaml-cpp)