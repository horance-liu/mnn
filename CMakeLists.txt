cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(mnn VERSION 1.0 LANGUAGES C CXX)

# Change the default build type from Debug to Release
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Build type: Debug. Performance will be terrible!")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

# Define user options
option(USE_SSE        "Build mnn with SSE library support"            ON)
option(USE_AVX        "Build mnn with AVX library support"            ON)
option(USE_AVX2       "Build mnn with AVX2 library support"           OFF)
option(USE_TBB        "Build mnn with TBB library support"            OFF)
option(USE_OMP        "Build mnn with OMP library support"            OFF)
option(USE_DOUBLE     "Build mnn with double precision computations"  OFF)

option(BUILD_TEST      "Set to ON to build tests"              ON)
option(BUILD_EXAMPLE   "Set to ON to build examples"           ON)

if(USE_DOUBLE)
    add_definitions(-DMNN_USE_DOUBLE)
endif()

# Find Intel Threading Building Blocks (TBB)
find_package(TBB QUIET)
if(USE_TBB AND TBB_FOUND)
    message(STATUS "Found Intel TBB: ${TBB_INCLUDE_DIR}")
    set(USE_OMP OFF)
    add_definitions(-DMNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    list(APPEND REQUIRED_LIBRARIES ${TBB_LIBRARIES})
elseif(USE_TBB AND NOT TBB_FOUND)
    message(FATAL_ERROR "Intel TBB not found. Please set TBB_INCLUDE_DIRS & "
            "TBB_LIBRARIES")
endif()

# Find Open Multi-Processing (OpenMP)
find_package(OpenMP QUIET)
if(USE_OMP AND OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    set(USE_TBB OFF)
    add_definitions(-DMNN_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
elseif(USE_OMP AND NOT OPENMP_FOUND)
    message(FATAL_ERROR "Can't find OpenMP. Please set OpenMP_C_FLAGS & "
            "OpenMP_CXX_FLAGS")
endif()

# in case that TBB and OMP are not enabled/found,
# we enable standard C++11 multithread support.
if((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))
    set(USE_PTHREAD ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    message(STATUS "TBB and OMP disabled: Using Pthread instead.")
else((NOT USE_TBB) AND (NOT USE_OMP))
    set(USE_PTHREAD OFF)
endif((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))

#
# Set c++ standard to c++14.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++14 support has been enabled by default.")

#
# Append compiler options for CPU ISA: SSE, AVX, AVX2
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)
    check_cxx_compiler_flag("-mfma"  COMPILER_HAS_AVX2_FLAG)

    # set Streaming SIMD Extension (SSE) instructions
    if(USE_SSE AND COMPILER_HAS_SSE_FLAG)
        add_definitions(-DMNN_USE_SSE)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
    endif(USE_SSE AND COMPILER_HAS_SSE_FLAG)

    # set Advanced Vector Extensions (AVX)
    if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
        add_definitions(-DMNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
    endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)
    
    # set Advanced Vector Extensions 2 (AVX2)
    if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
        add_definitions(-DMNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -mfma -march=core-avx2")
    endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)

    # include extra flags to the compiler
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing -Wno-deprecated")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pthread")
endif()

#
# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

#
# mnn target
file(GLOB_RECURSE mnn_srcs CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cc
    src/*.c
)

add_library(mnn ${mnn_srcs})

target_include_directories(mnn
    PUBLIC  include
    PRIVATE src
)

# Subdirectories for examples
if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif(BUILD_EXAMPLE)

# Subdirectories for tests
if(BUILD_TEST)
    add_subdirectory(test)
endif(BUILD_TEST)

# Summary
include(cmake/summary.cmake)
mnn_print_configuration_summary()
