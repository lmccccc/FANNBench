cmake_minimum_required (VERSION 2.8.12)
project(hnsw_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#   SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize")
#   else()
#     if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#       SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
#     elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#       SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
#     elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#       SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
#     endif()
# endif()

# add_executable(test_updates examples/updates_test.cpp)
# target_link_libraries(test_updates hnswlib)

# add_executable(searchKnnCloserFirst_test examples/searchKnnCloserFirst_test.cpp)
# target_link_libraries(searchKnnCloserFirst_test hnswlib)

# add_executable(main main.cpp sift_1b.cpp)
# target_link_libraries(main hnswlib)

# add_subdirectory(googletest)

include_directories(.)

file(GLOB TESTS tests/*.cpp)
FOREACH (path ${TESTS})
    get_filename_component(name ${path} NAME_WE)
    add_executable(${name} ${path})
ENDFOREACH ()



