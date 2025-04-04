cmake_minimum_required(VERSION 3.18)

project(pulp-nnx
  VERSION 0.3.0
  DESCRIPTION "Kernel library for PULP-based NN accelerators."
  LANGUAGES C)

add_library(pulp-nnx INTERFACE)

add_library(pulp-nnx-hal STATIC)

target_sources(pulp-nnx-hal PRIVATE util/pulp_nnx_util.c util/hwpe.c)
target_include_directories(pulp-nnx-hal PUBLIC inc util)

option(USE_NE16 "Use the NE16 accelerator.")
option(USE_NEUREKA "Use the N-EUREKA accelerator.")
option(USE_NEUREKA_V2 "Use the N-EUREKA v2 accelerator.")
option(ENABLE_BSP "Enable the build of the BSP for your chosen accelerator. Requires the PULP-SDK.")

if (NOT ${USE_NE16} AND NOT ${USE_NEUREKA} AND NOT ${USE_NEUREKA_V2})
  message(FATAL_ERROR "[PULP-NNX] No accelerator in use. Please set an appropriate USE_<acc> option.")
endif()

if(${ENABLE_BSP})
  add_library(pulp-nnx-bsp STATIC)
endif()

if (${USE_NE16})
  message(STATUS "[PULP-NNX] Using the NE16 accelerator.")
  target_sources(pulp-nnx-hal
    PRIVATE
    ne16/hal/ne16.c
    ne16/hal/ne16_task.c
  )
  target_include_directories(pulp-nnx-hal
    PUBLIC
    ne16/hal
    ne16/gvsoc
  )
  if(${ENABLE_BSP})
  target_sources(pulp-nnx-bsp
    PRIVATE
    ne16/bsp/ne16_pulp_bsp.c
    src/pulp_nnx_ne16.c
  )
  target_include_directories(pulp-nnx-bsp
    PUBLIC
    ne16/bsp
  )
  endif()

endif()

if (${USE_NEUREKA})
  message(STATUS "[PULP-NNX] Using the N-EUREKA accelerator.")
  target_sources(pulp-nnx-hal
    PRIVATE
    neureka/hal/neureka.c
    neureka/hal/neureka_task.c
  )
  target_include_directories(pulp-nnx-hal
    PUBLIC
    neureka/hal
    neureka/gvsoc
  )
  if(${ENABLE_BSP})
  target_sources(pulp-nnx-bsp
    PRIVATE
    neureka/bsp/neureka_siracusa_bsp.c
    src/pulp_nnx_neureka.c
  )
  target_include_directories(pulp-nnx-bsp
    PUBLIC
    neureka/bsp
  )
  endif()

endif()

if (${USE_NEUREKA_V2})
  message(STATUS "[PULP-NNX] Using the N-EUREKA v2 accelerator.")
  target_sources(pulp-nnx-hal
    PRIVATE
    neureka_v2/hal/neureka_v2.c
    neureka_v2/hal/neureka_v2_task.c
  )
  target_include_directories(pulp-nnx-hal
    PUBLIC
    neureka_v2/hal
    neureka_v2/gvsoc
  )
  if(${ENABLE_BSP})
  target_sources(pulp-nnx-bsp
    PRIVATE
    neureka_v2/bsp/neureka_v2_pulp_bsp.c
    src/pulp_nnx_neureka_v2.c
  )
  target_include_directories(pulp-nnx-bsp
    PUBLIC
    neureka_v2/bsp
  )
  endif()

endif()

target_link_libraries(pulp-nnx INTERFACE pulp-nnx-hal)
if(${ENABLE_BSP})
  target_link_libraries(pulp-nnx INTERFACE pulp-nnx-bsp)
endif()
