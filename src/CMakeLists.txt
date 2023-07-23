# include_directories(${Kokkos_INCLUDE_DIRS_RET}) # pass Kokkos include directories to our target application
add_executable(${PROJECT_NAME} "")
target_sources(${PROJECT_NAME}
  PUBLIC
  main.cpp
  BaseSolver.cpp
  global_setup.h
  global_class.h
  global_func.hpp
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  Kokkos::kokkos
)

# add_executable(${PROJECT_NAME} "")
# target_sources(${PROJECT_NAME}
#   PUBLIC
#   main.cpp
#   global_setup.h
# )
# add_library(cxxsrc
#   STATIC
#   global_class.h
#   BaseSolver.cpp
#   global_func.hpp
# )
# target_include_directories(cxxsrc
#   PUBLIC
#   ${Kokkos_INCLUDE_DIRS_RET})
# target_include_directories(${PROJECT_NAME}
#   PUBLIC
#   ${Kokkos_INCLUDE_DIRS_RET})
# target_link_libraries(${PROJECT_NAME}
#   PUBLIC
#   cxxsrc
#   Kokkos::kokkos
# )