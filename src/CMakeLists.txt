# include_directories(${Kokkos_INCLUDE_DIRS_RET})
add_executable(${PROJECT_NAME} "")
target_sources(${PROJECT_NAME}
  PUBLIC
  main.cpp
  BaseSolver.cpp
  global_setup.h
  global_class.h
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  Kokkos::kokkos
)