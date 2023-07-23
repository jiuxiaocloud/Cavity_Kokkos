set(Kokkos_CXX_STANDARD "17")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(Kokkos_ENABLE_DEBUG ON CACHE STRING "turn on Kokkos_Debug" FORCE)
    MESSAGE("-- ENABLE_KOKKOS_DEBUG=${Kokkos_ENABLE_DEBUG}")
endif(CMAKE_BUILD_TYPE MATCHES "Debug")

# set(Kokkos_ENABLE_TESTS ON CACHE STRING "turn on kokkos_tests" FORCE) #add compile time
# set(Kokkos_ENABLE_SERIAL OFF CACHE STRING "turn off Serial" FORCE)
set(${KOKKOS_ENABLE_BACKEND} ON CACHE STRING "turn on selected Kokkos backend" FORCE)
set(${Kokkos_BACKEND_ARCH} ON)
