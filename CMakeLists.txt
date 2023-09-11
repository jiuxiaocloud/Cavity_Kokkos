# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.17)

# default local cmake macro repository
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_COMPILER "${CMAKE_SOURCE_DIR}/external/kokkos/bin/nvcc_wrapper")
# some options in CMAKE && Kokkos need CACHE STRING FORCE set
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

PROJECT(CavityKokkos LANGUAGES CXX)
option(USE_DOUBLE "if double" ON)
option(USE_MPI "if enable mpi" ON)
option(AWARE_MPI "if enable AWARE_MPI" OFF)
option(Kokkos_ENABLE_TESTS "if enable Kokkos built-in test" OFF) # will highly increase compiling time cost
option(Kokkos_ENABLE_SERIAL "Open while no other backends opened" OFF)
option(Kokkos_ENABLE_OPENMP "if enable OMP Parallel" ON) # Host Parallel and Device Parallel can exist in the meantime
option(Kokkos_ENABLE_HIP "if enable Kokkos HIP backend" OFF)
option(Kokkos_ENABLE_CUDA "if enable Kokkos CUDA backend" ON)

# set(Kokkos_BACKEND_ARCH "Kokkos_ARCH_ZEN3") # Host ARCH supported by Kokkos
# set(Kokkos_BACKEND_ARCH "Kokkos_ARCH_AMPERE86") # CUDA ARCH supported by Kokkos
set(Kokkos_BACKEND_ARCH "Kokkos_ARCH_TURING75") # CUDA ARCH supported by Kokkos

set(INIPATH "${CMAKE_SOURCE_DIR}")
include(init_options)
IF(USE_MPI)
  add_subdirectory(external/mpiUtils)
ELSE()
ENDIF()

add_subdirectory(src)