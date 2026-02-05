cmake_minimum_required(VERSION 3.5)

project(FHE_RaiderSTREAM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to link against OpenFHE static libraries if needed.
option(BUILD_STATIC "Set to ON to use OpenFHE static libraries" OFF)

# ---- OpenFHE ----
find_package(OpenFHE CONFIG REQUIRED)
if(OpenFHE_FOUND)
  message(STATUS "Found OpenFHE")
  if(DEFINED BASE_OPENFHE_VERSION)
    message(STATUS "  OpenFHE Version: ${BASE_OPENFHE_VERSION}")
  endif()
  if(DEFINED OpenFHE_INCLUDE)
    message(STATUS "  OpenFHE include files: ${OpenFHE_INCLUDE}")
  endif()
  if(DEFINED OpenFHE_LIBDIR)
    message(STATUS "  OpenFHE lib files: ${OpenFHE_LIBDIR}")
  endif()
else()
  message(FATAL_ERROR "OpenFHE package not found!")
endif()

message(STATUS "OpenFHE_STATIC_LIBRARIES: ${OpenFHE_STATIC_LIBRARIES}")
message(STATUS "OpenFHE_SHARED_LIBRARIES: ${OpenFHE_SHARED_LIBRARIES}")

if(DEFINED OpenFHE_LIBDIR)
  link_directories(${OpenFHE_LIBDIR})
endif()

# ---- OpenMP ----
find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "OpenMP not found! Phase 1 requires OpenMP for multi-threading.")
endif()
message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")

# ---- Google Benchmark ----
# Prefer system package; fall back to FetchContent.
find_package(benchmark CONFIG QUIET)
if(NOT benchmark_FOUND)
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.5
  )
  FetchContent_MakeAvailable(benchmark)
endif()

add_executable(fhe_raiderstream
  src/main.cpp
  src/fixture.cpp
  src/kernels_sequential.cpp
)

# ---- Includes ----
target_include_directories(fhe_raiderstream
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# OpenFHE headers are commonly split across subdirs (core/pke/binfhe). Some
# headers include others without a prefix (e.g. openfhe.h -> openfhecore.h), so
# we add the common subdirs explicitly.
if(DEFINED OpenFHE_INCLUDE)
  target_include_directories(fhe_raiderstream
    PRIVATE
      ${OpenFHE_INCLUDE}
      ${OpenFHE_INCLUDE}/pke
      ${OpenFHE_INCLUDE}/core
      ${OpenFHE_INCLUDE}/binfhe
      ${OpenFHE_INCLUDE}/third-party/include
  )
endif()

# ---- Linking ----
# Some OpenFHE installs export a target; others only export variables.
set(_OPENFHE_TARGET "")
if(TARGET OpenFHE::openfhe)
  set(_OPENFHE_TARGET OpenFHE::openfhe)
elseif(TARGET OPENFHE::openfhe)
  set(_OPENFHE_TARGET OPENFHE::openfhe)
elseif(TARGET openfhe)
  set(_OPENFHE_TARGET openfhe)
endif()

if(NOT _OPENFHE_TARGET STREQUAL "")
  target_link_libraries(fhe_raiderstream PRIVATE ${_OPENFHE_TARGET})
else()
  if(BUILD_STATIC)
    if(NOT OpenFHE_STATIC_LIBRARIES)
      message(FATAL_ERROR "BUILD_STATIC is ON but OpenFHE_STATIC_LIBRARIES is empty")
    endif()
    target_link_libraries(fhe_raiderstream PRIVATE ${OpenFHE_STATIC_LIBRARIES})
  else()
    if(OpenFHE_SHARED_LIBRARIES)
      target_link_libraries(fhe_raiderstream PRIVATE ${OpenFHE_SHARED_LIBRARIES})
    elseif(OpenFHE_LIBRARIES)
      target_link_libraries(fhe_raiderstream PRIVATE ${OpenFHE_LIBRARIES})
    else()
      message(FATAL_ERROR "OpenFHE was found but no linkable targets/variables were provided.")
    endif()
  endif()
endif()

target_link_libraries(fhe_raiderstream PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)

# Ensure OpenMP compiler and linker flags are applied
if(OpenMP_CXX_FLAGS)
  target_compile_options(fhe_raiderstream PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_options(fhe_raiderstream PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Ensure OpenMP compiler and linker flags are applied
if(OpenMP_CXX_FLAGS)
  target_compile_options(fhe_raiderstream PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_options(fhe_raiderstream PRIVATE ${OpenMP_CXX_FLAGS})
endif()

message(STATUS "fhe_raiderstream configured successfully.")
