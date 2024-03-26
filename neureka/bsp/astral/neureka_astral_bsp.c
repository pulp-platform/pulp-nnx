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

#include "neureka_astral_bsp.h"
#include <pmsis.h>

#define NEUREKA_ASTRAL_CLUSTER_CTRL_BASE_ADDR (0x50200000)
#define NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_OFFS 0x18
#define NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR                                \
  (NEUREKA_ASTRAL_CLUSTER_CTRL_BASE_ADDR +                                   \
   NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_OFFS)
#define NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_CG_EN 0x800
#define NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_NEUREKA_SEL 0x2000
#define NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_HCI_PRIO 0x100
#define NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_HCI_MAXSTALL 0xff
#define NEUREKA_ASTRAL_MAX_STALL (8)
#define NEUREKA_ASTRAL_EVENT (1 << 12)
#define NEUREKA_ASTRAL_BASE_ADDR (0x50201000)

void neureka_astral_cg_enable() {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR |=
      NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_CG_EN;
}

void neureka_astral_cg_disable() {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR &=
      ~NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_CG_EN;
}

void neureka_astral_neureka_select() {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR |=
      NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_NEUREKA_SEL;
}

void neureka_astral_neureka_unselect() {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR &=
      ~NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_NEUREKA_SEL;
}

void neureka_astral_hci_setpriority_neureka() {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR |=
      NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_HCI_PRIO;
}

void neureka_astral_hci_setpriority_core() {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR &=
      ~NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_HCI_PRIO;
}

void neureka_astral_hci_reset_max_stall() {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR &=
      ~NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_HCI_MAXSTALL;
}

void neureka_astral_hci_set_max_stall(uint32_t max_stall) {
  *(volatile uint32_t *)NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_ADDR |=
      max_stall & NEUREKA_ASTRAL_CLUSTER_CTRL_HWPE_MASK_HCI_MAXSTALL;
}

void neureka_astral_open(neureka_astral_conf_t *conf) {
  neureka_astral_cg_enable();
  neureka_astral_neureka_select();
  neureka_astral_hci_setpriority_neureka();
  neureka_astral_hci_set_max_stall(conf->max_stall);
}

void neureka_astral_close() {
  neureka_astral_cg_disable();
  neureka_astral_neureka_unselect();
  neureka_astral_hci_reset_max_stall();
  neureka_astral_hci_setpriority_core();
}

void neureka_astral_event_wait_and_clear() {
  eu_evt_maskWaitAndClr(NEUREKA_ASTRAL_EVENT);
}

static const neureka_dev_t neureka_astral_dev = {
    .hwpe_dev = (struct hwpe_dev_t){
        .base_addr = (volatile uint32_t *)NEUREKA_ASTRAL_BASE_ADDR}};

const neureka_dev_t *neureka_astral_get_dev() {
  return &neureka_astral_dev;
}
