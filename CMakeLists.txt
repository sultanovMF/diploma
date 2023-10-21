cmake_minimum_required(VERSION 3.22)

project(viscowavesFD VERSION 0.0.1 DESCRIPTION "Finite difference modeling of viscoelastic waves propogation" LANGUAGES CXX)

# include(/home/murellos/vcpkg/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_CLANG_TIDY clang-tidy-17 -checks=-*,readability-*)
set(Boost_NO_WARN_NEW_VERSIONS 1)


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Compiler is Clang!")
endif()


# Add OpenMP to project
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(third_party/Eigen3/include/eigen3)


add_compile_options(-L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5)


add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen3/include)
#target_include_directories(Eigen3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen3/include/eigen3/unsupported/)

find_package(HDF5 REQUIRED COMPONENTS CXX)


add_library(HighFive INTERFACE)
target_include_directories(HighFive INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/HighFive/include)

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

file(GLOB HDF5_LIBRARIES "third_party/hdf5/libs/*")

add_library(HDF5 INTERFACE)
target_include_directories(HDF5 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hdf5)

add_library(tomlpp INTERFACE)
target_include_directories(tomlpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlpp/include)

add_executable(ssg_visco source/ssg_visco.cpp)
set_property(TARGET ssg_visco PROPERTY CXX_STANDARD 20)
target_link_libraries(ssg_visco Eigen3 HighFive tomlpp ${HDF5_LIBRARIES} )

file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})
