cmake_minimum_required(VERSION 3.16)
project(LeadLagAnalysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)  # CUDA 14 for better compatibility with 3.16
    add_definitions(-D__CUDACC__)
    
    # For CMake 3.16, use the older way to set CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -gencode=arch=compute_70,code=sm_70")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
    endif()
endif()

# Find packages
find_package(Threads REQUIRED)
find_package(OpenMP)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# SIMD flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/cpu/simd_correlation.cpp
    src/data/polygon_loader.cpp
    src/visualization/realtime_plot.cpp
)

# CUDA sources if available
if(CMAKE_CUDA_COMPILER)
    list(APPEND SOURCES src/cuda/lead_lag_kernel.cu)
    
    # For CMake 3.16, explicitly set CUDA properties
    set_source_files_properties(src/cuda/lead_lag_kernel.cu 
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endif()

# Create executable
add_executable(lead_lag_analyzer ${SOURCES})

# Set properties for CUDA if available (CMake 3.16 compatible way)
if(CMAKE_CUDA_COMPILER)
    # For CMake 3.16, we set CUDA flags differently
    target_compile_options(lead_lag_analyzer PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>
    )
    
    # Link CUDA libraries
    target_link_libraries(lead_lag_analyzer cudart)
endif()

# Link libraries
target_link_libraries(lead_lag_analyzer Threads::Threads)

if(OpenMP_FOUND)
    target_link_libraries(lead_lag_analyzer OpenMP::OpenMP_CXX)
endif()

# Optimizations for C++ code
target_compile_options(lead_lag_analyzer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
)

# Set C++ standard explicitly for better compatibility
set_target_properties(lead_lag_analyzer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Status messages
message(STATUS "Build configuration:")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  CUDA support: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  OpenMP support: ${OpenMP_FOUND}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "  CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif()
