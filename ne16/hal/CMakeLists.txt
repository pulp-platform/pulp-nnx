cmake_minimum_required(VERSION 3.18)

project(pulp-nnx-hal-ne16 LANGUAGES C)

add_library(pulp-nnx-hal-ne16
	STATIC
	ne16.c
	ne16_task.c
)

target_include_directories(pulp-nnx-hal-ne16 PUBLIC .)

add_subdirectory(../../util pulp-nnx-util)
target_link_libraries(pulp-nnx-hal-ne16 PUBLIC pulp-nnx-util)
