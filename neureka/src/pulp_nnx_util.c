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
 */

#include "pulp_nnx_util.h"
#include "pulp_nnx_hal.h"

void nnx_activate_gvsoc_logging(int log_level) {
  NEUREKA_WRITE_IO_REG(NEUREKA_REG_GVSOC_TRACE, log_level);
}

void nnx_deactivate_gvsoc_logging() {
  NEUREKA_WRITE_IO_REG(NEUREKA_REG_GVSOC_TRACE, 0);
}
