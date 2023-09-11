add_compile_options(-DUSE_MPI)

include_directories($ENV{MPI_PATH}/include)
find_library(MPI_CXX NAMES libmpi.so HINTS "$ENV{MPI_PATH}/lib")
message(STATUS "MPI settings: ")

IF(EXPLICIT_ALLOC)
  add_compile_options(-DEXPLICIT_ALLOC)

  IF(AWARE_MPI)
    add_compile_options(-DAWARE_MPI)
  ENDIF(AWARE_MPI)

  message(STATUS "  AWARE_MPI: ${AWARE_MPI}")
  message(STATUS "  MPI buffer allocate method: explicit")
ELSE()
  message(STATUS "  MPI buffer allocate method: implicit")
ENDIF()

message(STATUS "  MPI_HOME: $ENV{MPI_PATH}")
message(STATUS "  MPI_INC: $ENV{MPI_PATH}/include added")
message(STATUS "  MPI_CXX lib located: ${MPI_CXX} found")

include_directories(AFTER ${CMAKE_CURRENT_SOURCE_DIR})
add_library(mpiUtils "")

target_sources(mpiUtils
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/TestForException.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common_config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiEnums.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.h
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.h
)

target_link_libraries(mpiUtils
  PUBLIC
  ${MPI_CXX}
)