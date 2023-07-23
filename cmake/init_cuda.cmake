# find_package(NVHPC)#need add NVHPCConfig.cmake to /usr/lib/
find_package(CUDA)
enable_language(CUDA)
set(CUDA_INCLUDE_DIRS "$ENV{CUDA_PATH}/include") # CUDA_PATH in your bash must be exported
include_directories(${CUDA_INCLUDE_DIRS})
add_compile_options(-DENABLE_GPUA)
add_compile_options(-DENABLE_CUDA)

IF(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
ENDIF(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)