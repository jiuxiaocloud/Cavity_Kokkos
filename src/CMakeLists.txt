add_executable(${PROJECT_NAME} "")
target_sources(${PROJECT_NAME}
  PRIVATE
  main.cpp
  BaseSolver.cpp
  PUBLIC
  global_class.h
  global_setup.h
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  mpiUtils
  Kokkos::kokkos
)