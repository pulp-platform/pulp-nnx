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

#include "nnx_layer.h"
#include "pulp_nnx.h"

#include "ne16_gvsoc_logging.h"
#include "ne16_hal.h"

// Generated headers
#include "bias.h"
#include "input.h"
#include "layer_conf.h"
#include "output.h"
#include "scale.h"
#include "weight.h"

void execute_nnx_layer(void *unused_args) {
  ne16_activate_gvsoc_logging(NE16_GVSOC_LOG_LEVEL_ALL,
                              NE16_GVSOC_LOGGING_FORMAT_HEXADECIMAL);
  const int nnx_max_stall = 8;
  nnx_init(nnx_max_stall);

  nnx_task_t task;
  nnx_task_init(
      &task, WEIGHT_HEIGHT, GROUPS > 1, INPUT_BITS, OUTPUT_BITS, WEIGHT_BITS,
      weightOffsetModeLayerWise, WEIGHT_OFFSET,
      (nnx_quant_t){.shift_amount = OUTSHIFT,
                    .mode = quantMode8Bit,
                    .function =
                        HAS_RELU ? quantFunctionRelu : quantFunctionIdentity,
                    .flag_rounding = NE16_FLAG_UNUSED},
      (nnx_norm_t){.mode = normMode8Bit,
                   .flag_bias = HAS_BIAS ? NE16_FLAG_USED : NE16_FLAG_UNUSED,
                   .flag_shift = NE16_FLAG_UNUSED},
      STRIDE_HEIGHT);

  if (STRIDE_HEIGHT == 1) {
    nnx_task_set_dims(&task, INPUT_WIDTH, INPUT_CHANNEL, OUTPUT_HEIGHT,
                      OUTPUT_WIDTH, OUTPUT_CHANNEL, PADDING_TOP, PADDING_BOTTOM,
                      PADDING_RIGHT, PADDING_LEFT);
  } else {
    nnx_task_set_dims_stride2x2(&task, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL,
                                OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL,
                                WEIGHT_HEIGHT, WEIGHT_WIDTH, PADDING_TOP,
                                PADDING_BOTTOM, PADDING_RIGHT, PADDING_LEFT);
  }

  nnx_task_set_ptrs(&task, input, INPUT_WIDTH, INPUT_CHANNEL, INPUT_BITS,
                    PADDING_TOP, PADDING_LEFT, output, weight, scale, NULL,
#if HAS_BIAS == 1
                    bias
#else
                    NULL
#endif
  );

  nnx_dispatch_check_blocking();

  if (STRIDE_HEIGHT == 1) {
    nnx_dispatch_task(&task);
  } else {
    nnx_dispatch_task_stride2x2(&task, INPUT_WIDTH, INPUT_CHANNEL,
                                OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL,
                                WEIGHT_HEIGHT, WEIGHT_WIDTH);
  }
  // nnx_resolve_check_blocking(&task);
  while (!ne16_empty())
    ne16_event_wait();

  nnx_term();
  ne16_deactivate_gvsoc_logging();

  printf("\n");
  check_output();
}
