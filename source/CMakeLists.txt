# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
cmake_minimum_required(VERSION 3.5)  # 3.8 gives us built-in CUDA support

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe")
endif()
set(CMAKE_CXX_STANDARD 11)
project(GraphicsTF LANGUAGES C CXX CUDA)

add_definitions(-DGOOGLE_CUDA)

set(TFDR_INCLUDE_DIR	"${CMAKE_CURRENT_SOURCE_DIR}/include" 
						"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../.."
						"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
						"$ENV{TF_INC}"
						"$ENV{TF_INC}/external/nsync/public"
						"$ENV{TF_INC}/google/protobuf_archive/src"
						"$ENV{TF_INC}/external/eigen_archive"
						"$ENV{TF_INC}/tensorflow/include/external/com_google_absl")

file(GLOB_RECURSE cuda_srcs src/*.cu)
file(GLOB_RECURSE srcs src/*.cc)
file(GLOB_RECURSE inc include/*)
message(${cuda_srcs})
message(${srcs})
message(${inc})

link_directories($ENV{TF_LIB})
# set(TF_LIB_FILE "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(TF_LIB_FILE NAMES tensorflow_framework libtensorflow_framework.so.2 HINTS $ENV{TF_LIB})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    find_library(TF_LIB_FILE NAMES pywrap_tensorflow_internal _pywrap_tensorflow_internal HINTS $ENV{TF_LIB}/python)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Disable min and max function in Windows header to avoid compile ambiguous
    list(APPEND TFDR_CXX_COMPILE_OPTIONS -DNOMINMAX)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	list(APPEND TFDR_CXX_COMPILE_OPTIONS -D_GLIBCXX_USE_CXX11_ABI=0)
	list(APPEND TFDR_CUDA_COMPILE_OPTIONS -D_GLIBCXX_USE_CXX11_ABI=0)
	list(APPEND TFDR_CXX_COMPILE_OPTIONS -std=c++11)
	list(APPEND TFDR_CUDA_COMPILE_OPTIONS -std=c++11)
endif()
list(APPEND TFDR_CXX_COMPILE_OPTIONS -DNDEBUG)
list(APPEND TFDR_CUDA_COMPILE_OPTIONS -DNDEBUG)
list(APPEND TFDR_CUDA_COMPILE_OPTIONS --expt-relaxed-constexpr)

# include_directories(TFDR_INCLUDE_DIR)

add_library(graphicstf SHARED ${cuda_srcs} ${srcs} ${inc})
target_include_directories(graphicstf PRIVATE ${TFDR_INCLUDE_DIR})
target_link_libraries(graphicstf PRIVATE ${TF_LIB_FILE})
target_compile_options(graphicstf PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${TFDR_CXX_COMPILE_OPTIONS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${TFDR_CUDA_COMPILE_OPTIONS}>)

# Put the compiled library in the python package folder, rather than whatever build folder is being used
install(TARGETS graphicstf DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
