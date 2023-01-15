cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(${THIS} pytorch-exp-app)
set(NET_LIB net_lib)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_PREFIX_PATH "C:\\Users\\user\\Repos\\C++Libraries\\libtorch")

project(pytorch-exp-app C CXX)
set(OpenCV_DIR "C:\\Users\\user\\Repos\\C++Libraries\\opencv\\build\\x64\\vc16")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(app)
add_subdirectory(nets)
add_subdirectory(pymodel_to_cpp)
add_subdirectory(opencv_and_torch)




