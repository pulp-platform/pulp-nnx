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

#ifndef __NE16_GVSOC_LOGGING_H__
#define __NE16_GVSOC_LOGGING_H__

#include "ne16_hal.h"

typedef enum ne16_gvsoc_logging_format_e {
  NE16_GVSOC_LOGGING_FORMAT_DECIMAL = 0,
  NE16_GVSOC_LOGGING_FORMAT_HEXADECIMAL = 3
} ne16_gvsoc_logging_format_e;

typedef enum ne16_gvsoc_log_level_e {
  NE16_GVSOC_LOG_LEVEL_CONFIG = 0,
  NE16_GVSOC_LOG_LEVEL_ACTIV_INOUT = 1,
  NE16_GVSOC_LOG_LEVEL_DEBUG = 2,
  NE16_GVSOC_LOG_LEVEL_ALL = 3
} ne16_gvsoc_log_level_e;

static inline void
ne16_activate_gvsoc_logging(ne16_gvsoc_log_level_e log_level,
                            ne16_gvsoc_logging_format_e format) {
  NE16_WRITE_IO_REG(sizeof(nnx_task_data_t), log_level);
  NE16_WRITE_IO_REG(sizeof(nnx_task_data_t) + 4, format);
}

static inline void ne16_deactivate_gvsoc_logging() {
  NE16_WRITE_IO_REG(sizeof(nnx_task_data_t), 0);
}

#endif // __NE16_GVSOC_LOGGING_H__
