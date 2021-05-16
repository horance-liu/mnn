####
# Note: If you use 2.6 or 2.4, God kills a kitten. Seriously.
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

####
# Set variables:
#   * PROJECT_NAME
#   * PROJECT_VERSION
project(mnn VERSION 1.0 LANGUAGES C CXX)

#####
# Enables link_directories() treat paths relative
# to the source dir.
if(POLICY CMP0015)
    cmake_policy(SET CMP0015 NEW)
endif(POLICY CMP0015)

#####
# Enables project() command manages VERSION variables.
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

#####
# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

####
# Define user options
option(USE_SSE        "Build mnn with SSE library support"            ON)
option(USE_AVX        "Build mnn with AVX library support"            ON)
option(USE_AVX2       "Build mnn with AVX2 library support"           OFF)
option(USE_TBB        "Build mnn with TBB library support"            OFF)
option(USE_OMP        "Build mnn with OMP library support"            OFF)
option(USE_SERIALIZER "Build mnn with Serialization support"          ON)
option(USE_DOUBLE     "Build mnn with double precision computations"  OFF)

option(BUILD_TEST      "Set to ON to build tests"              ON)
option(BUILD_EXAMPLE   "Set to ON to build examples"           ON)
option(BUILD_BENCHMARK "Set to ON to build benchmarks"         ON)
option(ENABLE_PROFILE  "Set to ON to build with profiling"     OFF)

####
# Global Defines
if(NOT USE_SERIALIZER)
    add_definitions(-DMNN_NO_SERIALIZATION)
endif()

if(USE_DOUBLE)
    add_definitions(-DMNN_USE_DOUBLE)
endif()

####
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# mnn provides a couple of multithreading solutions.
# The user can specify to use Intel Threading Building Blocks (TBB)
# or Open Multi-Processing (OpenMP) as a backend for multi threading
# processing. In case that none of this libraries are required, mnn
# will use the standard C++11 Thread support library.

# Find Intel Threading Building Blocks (TBB)
find_package(TBB QUIET)
if(USE_TBB AND TBB_FOUND)
    message(STATUS "Found Intel TBB: ${TBB_INCLUDE_DIR}")
    # In case that TBB is found we force to disable OpenMP since
    # mnn does not support mutiple multithreading backends.
    set(USE_OMP OFF)
    #TODO: add definitions in configure
    add_definitions(-DMNN_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})
    list(APPEND REQUIRED_LIBRARIES ${TBB_LIBRARIES})
elseif(USE_TBB AND NOT TBB_FOUND)
    # In case the user sets the flag USE_TBB to ON, the CMake build-tree
    # will require to find TBB in your system. Otherwise, the user can
    # set the paths to headers and libs by hand.
    message(FATAL_ERROR "Intel TBB not found. Please set TBB_INCLUDE_DIRS & "
            "TBB_LIBRARIES")
endif()

# Find Open Multi-Processing (OpenMP)
find_package(OpenMP QUIET)
if(USE_OMP AND OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    # In case that OMP is found we force to disable Intel TBB since
    # mnn does not support mutiple multithreading backends.
    set(USE_TBB OFF)
    add_definitions(-DMNN_USE_OMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
elseif(USE_OMP AND NOT OPENMP_FOUND)
    # In case the user sets the flag USE_OMP to ON, the CMake build-tree
    # will require to find OMP in your system. Otherwise, the user can
    # set the CMAKE_C_FLAGS and CMAKE_CXX_FLAGS by hand.
    message(FATAL_ERROR "Can't find OpenMP. Please set OpenMP_C_FLAGS & "
            "OpenMP_CXX_FLAGS")
endif()

# in case that TBB and OMP are not enabled/found,
# we enable standard C++11 multithread support.
if((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))
    #list(APPEND EXTRA_C_FLAGS -pthread)
    set(USE_PTHREAD ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    message(STATUS "TBB and OMP disabled: Using Pthread instead.")
else((NOT USE_TBB) AND (NOT USE_OMP))
    set(USE_PTHREAD OFF)
endif((NOT USE_TBB) AND (NOT USE_OMP) AND (NOT WIN32))


####
# Setup the compiler options
# set c++ standard to c++14.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++14 support has been enabled by default.")

####
# Append compiler options for CPU ISA.
# SSE, AVX, AVX2
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

####
# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

if(ENABLE_PROFILE AND (CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -pg")
endif()

####
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

# Subdirectories for examples, test
if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif(BUILD_EXAMPLE)

if(BUILD_TEST)
    add_subdirectory(test)
endif(BUILD_TEST)


# Configuration summary
include(cmake/summary.cmake)
mnn_print_configuration_summary()


