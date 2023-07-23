# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.17)

# default local cmake macro repository
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# some options in CMAKE && Kokkos need CACHE STRING FORCE set
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

PROJECT(LargeMartix LANGUAGES CXX)
option(USE_DOUBLE "if double" ON)
option(ENABLE_KOKKOS_OPENMP "if need kokkos midleware" ON)
option(ENABLE_KOKKOS_CUDA "if enable cuda,OFF when ENABLE_HIP" OFF)
option(ENABLE_KOKKOS_HIP "if enable hip, OFF when ENABLE_CUDA" OFF)
option(USE_MPI "if enable mpi" OFF)
option(AWARE_MPI "if enable AWARE_MPI" OFF)

include(init_options)
add_subdirectory(src)
