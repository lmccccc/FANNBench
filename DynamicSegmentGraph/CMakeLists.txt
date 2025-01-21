cmake_minimum_required(VERSION 3.12)
project(RangeFiltering-ANNS)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "Found OpenMP")
else()
    message(FATAL_ERROR "Could NOT find OpenMP")
endif()

message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_STANDARD 11)
message(${CMAKE_SYSTEM})

find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)
message(STATUS ${Boost_LIBRARIES})
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    SET (LD_LIBRARY_PATH ${BOOST_LIBRARYDIR})
else()
    message(FATAL_ERROR "Could NOT find Boost")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/common)
include_directories(${PROJECT_SOURCE_DIR}/src)


if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
   set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
   message(STATUS "Debug Mode")
else()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2 -DUSE_SSE")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
endif()


set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NOSIMD "build hnsw on SIMD" ON)

if(NOSIMD)
    add_definitions(-DNO_MANUAL_VECTORIZATION)
    add_definitions(-DKGRAPH_NO_VECTORIZE)
endif()

add_subdirectory(src/common)
add_subdirectory(benchmark)
