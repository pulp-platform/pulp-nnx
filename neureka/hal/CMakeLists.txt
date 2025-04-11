cmake_minimum_required(VERSION 3.18)

project(pulp-nnx-hal-neureka LANGUAGES C)

add_library(pulp-nnx-hal-neureka
	STATIC
	neureka.c
	neureka_task.c
)

target_include_directories(pulp-nnx-hal-neureka PUBLIC .)

add_subdirectory(../../util pulp-nnx-util)
target_link_libraries(pulp-nnx-hal-neureka PUBLIC pulp-nnx-util)
