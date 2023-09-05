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

APP := main

LIBDIR := $(abspath ../..)


# Include directories

## Test
INC_DIRS += inc gen_inc

## PULP-NNX
INC_DIRS += $(LIBDIR)/inc

## NE16
INC_DIRS += $(LIBDIR)/ne16/hal $(LIBDIR)/ne16/gvsoc

## Util
INC_DIRS += $(LIBDIR)/util

INC_FLAGS += $(addprefix -I,$(INC_DIRS))


# Source files

## Test
APP_SRCS += $(wildcard src/*.c)

## PULP-NNX
APP_SRCS += $(LIBDIR)/src/pulp_nnx_ne16.c

## NE16
APP_SRCS += $(wildcard $(LIBDIR)/ne16/hal/*.c) $(wildcard $(LIBDIR)/ne16/gvsoc/*.c)

## Util
APP_SRCS += $(LIBDIR)/util/pulp_nnx_util.c


APP_CFLAGS += $(INC_FLAGS) -O2 -w

include $(RULES_DIR)/pmsis_rules.mk
