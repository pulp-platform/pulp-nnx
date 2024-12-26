/*
 * Luka Macan <luka.macan@unibo.it>
 *
 * Copyright 2023 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __NEUREKA_V2_GVSOC_H__
#define __NEUREKA_V2_GVSOC_H__

#include "neureka_v2.h"
#include "neureka_v2_task.h"

#define NEUREKA_V2_REG_GVSOC_LOG_LEVEL 25
#define NEUREKA_V2_REG_GVSOC_LOG_FORMAT 26
#define NEUREKA_V2_REG_GVSOC_FRAME_REG 27

typedef enum neureka_v2_gvsoc_log_format_e {
  NEUREKA_V2_GVSOC_LOG_FORMAT_DECIMAL = 0,
  NEUREKA_V2_GVSOC_LOG_FORMAT_HEXADECIMAL = 3
} neureka_v2_gvsoc_log_format_e;

typedef enum neureka_v2_gvsoc_log_level_e {
  NEUREKA_V2_GVSOC_LOG_LEVEL_JOB_START_END = 0,
  NEUREKA_V2_GVSOC_LOG_LEVEL_CONFIG = 1,
  NEUREKA_V2_GVSOC_LOG_LEVEL_ACTIV_INOUT = 2,
  NEUREKA_V2_GVSOC_LOG_LEVEL_ALL = 3
} neureka_v2_gvsoc_log_level_e;

static void
neureka_v2_gvsoc_log_activate(const neureka_v2_dev_t *dev,
                              neureka_v2_gvsoc_log_level_e log_level,
                              neureka_v2_gvsoc_log_format_e format) {
  hwpe_task_reg_write(&dev->hwpe_dev, NEUREKA_V2_REG_GVSOC_LOG_LEVEL,
                      log_level);
  hwpe_task_reg_write(&dev->hwpe_dev, NEUREKA_V2_REG_GVSOC_LOG_FORMAT, format);
}

static void neureka_v2_gvsoc_log_deactivate(const neureka_v2_dev_t *dev) {
  hwpe_task_reg_write(&dev->hwpe_dev, NEUREKA_V2_REG_GVSOC_LOG_LEVEL,
                      NEUREKA_V2_GVSOC_LOG_LEVEL_JOB_START_END);
}

#endif // __NEUREKA_V2_GVSOC_H__
