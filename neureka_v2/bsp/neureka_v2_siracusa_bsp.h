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

#ifndef __NEUREKA_V2_SIRACUSA_BSP_H__
#define __NEUREKA_V2_SIRACUSA_BSP_H__

#include "neureka_v2.h"
#include <stdint.h>

/**
 * neureka_v2_siracusa_setpriority_neureka_v2
 *
 * Set HCI interconnect bus priority to prioritize neureka_v2.
 */
void neureka_v2_siracusa_hci_setpriority_neureka_v2();

/**
 * neureka_v2_siracusa_setpriority_core
 *
 * Set HCI bus priority to prioritize cores.
 */
void neureka_v2_siracusa_hci_setpriority_core();

/**
 * neureka_v2_siracusa_hci_reset_maxstall
 *
 * Reset the HCI bus maxstall parameter.
 * TODO: Check if it disables it also or just resets?
 */
void neureka_v2_siracusa_hci_reset_max_stall();

/**
 * neureka_v2_siracusa_hci_set_maxstall
 *
 * Set the HCI bus maxstall. Maxstall defines how many cycles
 * will the HCI bus stall the lower priority master, i.e. neureka_v2 or core,
 * before letting it do a transaction.
 */
void neureka_v2_siracusa_hci_set_max_stall(uint32_t max_stall);

typedef struct neureka_v2_siracusa_conf_t {
  int max_stall;
} neureka_v2_siracusa_conf_t;

void neureka_v2_siracusa_open(neureka_v2_siracusa_conf_t *conf);
void neureka_v2_siracusa_close();
void neureka_v2_siracusa_event_wait_and_clear();
const neureka_v2_dev_t *neureka_v2_siracusa_get_dev();

#endif // !__NEUREKA_V2_SIRACUSA_BSP_H__
