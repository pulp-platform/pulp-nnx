# Luka Macan <luka.macan@unibo.it>
#
# Copyright 2023 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Set runner_args="--trace=ne16" if you want to trace what is happening in the ne16

ACCELERATOR ?= ne16

APP := main
LIBDIR := $(abspath ../../..)
ACC_DIR := $(LIBDIR)/$(ACCELERATOR)


# Include directories

## Test
INC_DIRS += inc

## Library
INC_DIRS += $(LIBDIR)/inc $(LIBDIR)/util

## Accelerator
INC_DIRS += $(ACC_DIR)/hal $(ACC_DIR)/gvsoc $(ACC_DIR)/bsp

## Generated 
INC_DIRS += gen/inc

INC_FLAGS += $(addprefix -I,$(INC_DIRS))
APP_CFLAGS += $(INC_FLAGS)


# Source files

## Test
APP_SRCS += $(wildcard src/*.c)

## Library
APP_SRCS += $(LIBDIR)/src/pulp_nnx_$(ACCELERATOR).c $(wildcard $(LIBDIR)/util/*.c)

## Accelerator
APP_SRCS += $(wildcard $(ACC_DIR)/hal/*.c) $(wildcard $(ACC_DIR)/gvsoc/*.c) $(wildcard $(ACC_DIR)/bsp/*.c)

## Generated 
APP_SRCS += $(wildcard gen/src/*.c)


# Flags

ACCELERATOR_UPPERCASE := $(shell echo $(ACCELERATOR) | tr [:lower:] [:upper:])
APP_CFLAGS += -DNNX_ACCELERATOR=\"$(ACCELERATOR)\" -DNNX_$(ACCELERATOR_UPPERCASE)

APP_CFLAGS += -O2 -w -Wall -Werror

include $(RULES_DIR)/pmsis_rules.mk
