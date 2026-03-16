cmake_minimum_required(VERSION 3.5)

project(FHE_RaiderSTREAM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Option to link against OpenFHE static libraries if needed.
option(BUILD_STATIC "Set to ON to use OpenFHE static libraries" OFF)
option(ENABLE_DCRT "Build the DCRTPoly backend" ON)
option(ENABLE_CIPHERTEXT "Build the Ciphertext backend" OFF)

# ---- intel hexl ----
find_package(HEXL CONFIG)
if(HEXL_FOUND)
else()
  message(STATUS "Intel HEXL not found; won't use")
endif()

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
  src/common/BenchmarkUtils.cpp
)

# ---- MPI (Optional) ----
option(ENABLE_MPI "Build with MPI support" ON)
if(ENABLE_MPI)
  find_package(MPI QUIET)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_LIBRARIES}")
    target_compile_definitions(fhe_raiderstream PRIVATE RAIDERSTREAM_MPI)
    target_link_libraries(fhe_raiderstream PRIVATE MPI::MPI_CXX)
  else()
    message(STATUS "MPI not found. Distributed benchmarking disabled.")
  endif()
endif()

if(ENABLE_DCRT)
  target_sources(fhe_raiderstream PRIVATE
    src/backends/dcrt/DCRTFixture.cpp
  )

  # Support either a monolithic DCRT kernel file or split kernel files.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/dcrt/kernels_dcrt.cpp")
    target_sources(fhe_raiderstream PRIVATE
      src/backends/dcrt/kernels_dcrt.cpp
    )
    message(STATUS "Using monolithic DCRT kernels file: kernels_dcrt.cpp")
  else()
    target_sources(fhe_raiderstream PRIVATE
      src/backends/dcrt/kernels_sequential.cpp
      src/backends/dcrt/kernels_gather.cpp
      src/backends/dcrt/kernels_scatter.cpp
      src/backends/dcrt/kernels_scatter_gather.cpp
      src/backends/dcrt/kernels_ntt.cpp
    )
    message(STATUS "Using split DCRT kernel files")
  endif()

  target_compile_definitions(fhe_raiderstream PRIVATE RAIDERSTREAM_DCRT)
  message(STATUS "DCRT backend enabled")
endif()

if(ENABLE_CIPHERTEXT)
  target_sources(fhe_raiderstream PRIVATE
    src/backends/ciphertext/CTFixture.cpp
    src/backends/ciphertext/kernels_ct.cpp
  )
  target_compile_definitions(fhe_raiderstream PRIVATE RAIDERSTREAM_CIPHERTEXT)
  message(STATUS "Ciphertext backend enabled")
endif()

if(NOT ENABLE_DCRT AND NOT ENABLE_CIPHERTEXT)
  message(WARNING "No backend enabled. Enable at least one backend (ENABLE_DCRT or ENABLE_CIPHERTEXT).")
endif()

# ---- Includes ----
target_include_directories(fhe_raiderstream
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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

message(STATUS "fhe_raiderstream configured successfully.")
