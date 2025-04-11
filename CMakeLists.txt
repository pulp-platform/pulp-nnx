cmake_minimum_required(VERSION 3.18)

project(pulp-nnx
	VERSION 0.3.0
	DESCRIPTION "Kernel library for PULP-based NN accelerators."
	LANGUAGES C)

add_library(pulp-nnx STATIC)

target_include_directories(pulp-nnx PUBLIC inc)

option(USE_NE16 "Use the NE16 accelerator.")
option(USE_NEUREKA "Use the N-EUREKA accelerator.")
option(USE_NEUREKA_V2 "Use the N-EUREKA v2 accelerator.")

if (NOT ${USE_NE16} AND NOT ${USE_NEUREKA} AND NOT ${USE_NEUREKA_V2})
	message(FATAL_ERROR "[PULP-NNX] No accelerator in use. Please set an appropriate USE_<acc> option.")
endif()

if (${USE_NE16})
	message(STATUS "[PULP-NNX] Using the NE16 accelerator.")
	add_subdirectory(ne16/hal)
	target_link_libraries(pulp-nnx PUBLIC pulp-nnx-hal-ne16)
	target_sources(pulp-nnx
		PRIVATE
			ne16/bsp/ne16_pulp_bsp.c
			ne16/hal/ne16.c
			ne16/hal/ne16_task.c
			src/pulp_nnx_ne16.c
		)
	target_include_directories(pulp-nnx
		PUBLIC
			ne16/bsp
			ne16/hal
			ne16/gvsoc
		)
endif()

if (${USE_NEUREKA})
	message(STATUS "[PULP-NNX] Using the N-EUREKA accelerator.")
	add_subdirectory(neureka/hal)
	target_link_libraries(pulp-nnx PUBLIC pulp-nnx-hal-neureka)
	target_sources(pulp-nnx
		PRIVATE
			neureka/bsp/neureka_siracusa_bsp.c
			src/pulp_nnx_neureka.c
		)
	target_include_directories(pulp-nnx
		PUBLIC
			neureka/bsp
			neureka/gvsoc
		)
endif()

if (${USE_NEUREKA_V2})
	message(STATUS "[PULP-NNX] Using the N-EUREKA v2 accelerator.")
	add_subdirectory(neureka_v2/hal)
	target_link_libraries(pulp-nnx PUBLIC pulp-nnx-hal-neureka-v2)
	target_sources(pulp-nnx
		PRIVATE
			neureka_v2/bsp/neureka_v2_siracusa_bsp.c
			src/pulp_nnx_neureka_v2.c
		)
	target_include_directories(pulp-nnx
		PUBLIC
			neureka_v2/bsp
			neureka_v2/gvsoc
		)
endif()
