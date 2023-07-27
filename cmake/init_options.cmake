# ######################## KOKKOS ###################################
# #for use kokkos 1.add_subdirectory(path/to/kokkos) 2.include_directories 3.target_link_libraries(kokkos)
include(init_kokkos)
set(KOKKOS_SUBMOD_LOCATION "${CMAKE_SOURCE_DIR}/external/kokkos")
add_subdirectory(${KOKKOS_SUBMOD_LOCATION})
# ######################## KOKKOS ###################################
IF(USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE) # 将参数从cmakelist传入程序中
ENDIF(USE_DOUBLE)

add_compile_options(-DIniFile="${INIPATH}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
message(STATUS "CMAKE STATUS:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "  CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
