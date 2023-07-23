# ######################## KOKKOS ###################################
# #for use kokkos 1.add_subdirectory(path/to/kokkos) 2.include_directories 3.target_link_libraries(kokkos)
message(STATUS "KOKKOS Configure Section::")
set(KOKKOS_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/external/kokkos")
include(init_kokkos)
add_subdirectory(${KOKKOS_SUBMOD_LOCATION})
# ######################## KOKKOS ###################################
IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE) # 将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)