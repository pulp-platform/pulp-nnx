cmake_minimum_required(VERSION 3.18)

# This is to omit the cross-compilation test
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

project(test-pulp-nnx
	LANGUAGES C ASM)

add_executable(test-pulp-nnx)

file(GLOB gen_srcs gen/src/*.c)
set(app_srcs src/main.c src/nnx_layer.c)

target_sources(test-pulp-nnx PRIVATE ${app_srcs} ${gen_srcs})
target_include_directories(test-pulp-nnx PRIVATE inc gen/inc)

set(NUM_CORES 8 CACHE STRING "Set the number of cores used. Default 8")
set(ACCELERATOR NE16 CACHE STRING "Choose an accelerator to compile the library for. Default NE16")
set_property(CACHE ACCELERATOR PROPERTY STRINGS NE16 NEUREKA)

# TODO: should this go into pulp-sdk
add_compile_options(-DNUM_CORES=${NUM_CORES})

include(cmake/pulp-sdk-siracusa.cmake)
target_link_libraries(test-pulp-nnx PRIVATE pulp-sdk)

if (${ACCELERATOR} STREQUAL NE16)
	set(USE_NE16 ON)
	target_compile_definitions(test-pulp-nnx PRIVATE NNX_ACCELERATOR="ne16" NNX_NE16)
elseif(${ACCELERATOR} STREQUAL NEUREKA)
	set(USE_NEUREKA ON)
	target_compile_definitions(test-pulp-nnx PRIVATE NNX_ACCELERATOR="neureka" NNX_NEUREKA)
endif()

add_subdirectory(../.. pulp-nnx)
target_link_libraries(pulp-nnx PUBLIC pulp-sdk)
target_include_directories(pulp-nnx SYSTEM PUBLIC ${PULP_SDK_INCLUDES})
target_link_libraries(test-pulp-nnx PRIVATE pulp-nnx)
