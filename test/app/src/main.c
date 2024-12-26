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

#include <pmsis.h>

#include "layer_util.h"
#include "nnx_layer.h"
#include "output.h"
#ifdef NNX_NEUREKA_V2
#include "string.h"
#include "weight.h"
#include "weight_l2.h"
#endif

int main() {
  struct pi_device cl_dev;
  struct pi_cluster_conf cl_conf;
  struct pi_cluster_task cl_task;

  printf("\nTest " TEST_NAME " starting\n");

  printf("\nAccelerator: " NNX_ACCELERATOR "\n");

  printf("\n");
  layer_info();

#ifdef NNX_NEUREKA_V2
  // We have to initialize the mram/sram weight memory from l2
  memcpy((void *)weight, (void *)weight_l2, WEIGHT_SIZE);
#endif

  pi_cluster_conf_init(&cl_conf);
  pi_open_from_conf(&cl_dev, &cl_conf);
  if (pi_cluster_open(&cl_dev)) {
    printf("ERROR: Failed to open cluster.\n");
    pmsis_exit(-1);
  }
  pi_cluster_send_task_to_cl(
      &cl_dev, pi_cluster_task(&cl_task, execute_nnx_layer, NULL));

  printf("\n");
  check_output();

  pi_cluster_close(&cl_dev);

  printf("\nTest " TEST_NAME " finished\n");

  return 0;
}
