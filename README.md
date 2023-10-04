# Cavity_Kokkos
- This is a Kokkos-based implementation of the cavity flow solver.

## 1.Dependencies 
- gcc, g++ suit with bundled openmp on Unix
- common mpi libs on Unix
- CUDA for NV GPUs or ROCm for AMD GPUs for Heterogeneous acceleration on Unix

## 2.Tested environment 
- Ubuntu 20.04 latest version

## 3.Build

````bash
	$ cd ${workspaceFolder}/build && cmake .. && make -j
````
- 1. exectuable program file is located in ${workspaceFolder}/build/src/cavity_kokkos
- 2. cmake build Release version for CUDA acceleration by default, use cmake -DCMAKE_BUILD_TYPE=Debug to build Debug version, Debug version use CPU serial Backends.
- 3. set options of ${workspaceFolder}/CMakeLists.txt for different accleration backends.

## 4.Run
- 1. Cavity read input.txt all the time, but you can still append mesh size arguments in CMD, it's useful while mpi is running
- 2. mpi running:
	````bash
		$ mpirun -n 2 ./build/src/Cavity 256 128
	````
	is the same as ./build/src/Cavity and ./build/src/Cavity 256 256