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

#include "neureka_v2.h"

#define NEUREKA_V2_STATUS_EMPTY (0x000)
#define NEUREKA_V2_STATUS_FULL (0x101)

inline int neureka_v2_task_queue_tasks_in_flight(const neureka_v2_dev_t *dev) {
  uint32_t status = hwpe_task_queue_status(&dev->hwpe_dev);
  return (status & 0x1) + ((status >> 8) & 0x1);
}

inline int neureka_v2_task_queue_empty(const neureka_v2_dev_t *dev) {
  return hwpe_task_queue_status(&dev->hwpe_dev) == NEUREKA_V2_STATUS_EMPTY;
}

inline int neureka_v2_task_queue_full(const neureka_v2_dev_t *dev) {
  return hwpe_task_queue_status(&dev->hwpe_dev) == NEUREKA_V2_STATUS_FULL;
}
