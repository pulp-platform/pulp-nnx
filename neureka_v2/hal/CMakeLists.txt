cmake_minimum_required(VERSION 3.18)

project(pulp-nnx-hal-neureka-v2 LANGUAGES C)

add_library(pulp-nnx-hal-neureka-v2
	STATIC
	neureka_v2.c
	neureka_v2_task.c
)

target_include_directories(pulp-nnx-hal-neureka-v2 PUBLIC .)

add_subdirectory(../../util pulp-nnx-util)
target_link_libraries(pulp-nnx-hal-neureka-v2 PUBLIC pulp-nnx-util)
