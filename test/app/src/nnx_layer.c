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

#include "nnx_layer.h"
#include "ne16.h"
#include "ne16_gvsoc.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "pulp_nnx_ne16.h"
#include <pmsis.h>

// Generated headers
#include "bias.h"
#include "input.h"
#include "layer_conf.h"
#include "output.h"
#include "scale.h"
#include "weight.h"

static void task_prepare(ne16_task_t *task) {
  ne16_task_init(task, WEIGHT_HEIGHT, GROUPS > 1, INPUT_BITS, OUTPUT_BITS,
                 WEIGHT_BITS, weightOffsetModeLayerWise, WEIGHT_OFFSET,
                 (ne16_quant_t){.shift_amount = OUTSHIFT,
                                .mode = quantMode8Bit,
                                .function = HAS_RELU ? quantFunctionRelu
                                                     : quantFunctionIdentity,
                                .flag_rounding = ne16TaskFlagFalse},
                 (ne16_norm_t){.mode = normMode8Bit,
                               .flag_bias = HAS_BIAS ? ne16TaskFlagTrue
                                                     : ne16TaskFlagFalse,
                               .flag_shift = ne16TaskFlagFalse},
                 STRIDE_HEIGHT);

  if (STRIDE_WIDTH == 2 && STRIDE_HEIGHT == 2) {
    ne16_task_set_dims_stride2x2(
        task, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, INPUT_WIDTH,
        INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL,
        OUTPUT_WIDTH, OUTPUT_CHANNEL, WEIGHT_HEIGHT, WEIGHT_WIDTH, PADDING_TOP,
        PADDING_BOTTOM, PADDING_RIGHT, PADDING_LEFT);
  } else {
    ne16_task_set_dims(task, INPUT_WIDTH, INPUT_CHANNEL, INPUT_WIDTH,
                       INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH,
                       OUTPUT_CHANNEL, OUTPUT_WIDTH, OUTPUT_CHANNEL, PADDING_TOP,
                       PADDING_BOTTOM, PADDING_RIGHT, PADDING_LEFT);
  }

  ne16_task_set_ptrs(task, (uint32_t)input, INPUT_WIDTH, INPUT_CHANNEL,
                     INPUT_BITS, PADDING_TOP, PADDING_LEFT, (uint32_t)output,
                     (uint32_t)weight, (uint32_t)scale, NULL,
#if HAS_BIAS == 1
                     (uint32_t)bias
#else
                     NULL
#endif
  );
}

static void task_execute(ne16_task_t *task) {
  ne16_dev_t *dev = ne16_pulp_get_dev();

  ne16_gvsoc_log_activate(dev, NE16_GVSOC_LOG_LEVEL_CONFIG,
                          NE16_GVSOC_LOG_FORMAT_HEXADECIMAL);

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(dev, &conf);

  ne16_nnx_dispatch_wait(dev);

  if (STRIDE_WIDTH == 2 && STRIDE_HEIGHT == 2) {
    ne16_nnx_dispatch_stride2x2(dev, task, INPUT_WIDTH, INPUT_CHANNEL, INPUT_WIDTH,
                                INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH,
                                OUTPUT_CHANNEL, OUTPUT_WIDTH, OUTPUT_CHANNEL,
                                WEIGHT_HEIGHT, WEIGHT_WIDTH);
  } else {
    ne16_nnx_dispatch(dev, task);
  }

  ne16_nnx_resolve_wait(dev, task);

  ne16_nnx_term(dev);

  ne16_gvsoc_log_deactivate(dev);
}

void execute_nnx_layer(void *args) {
  ne16_task_t task;
  task_prepare(&task);
  task_execute(&task);
}
