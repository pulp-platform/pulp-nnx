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
#include <pmsis.h>

#ifdef NNX_NE16

#include "ne16.h"
#include "ne16_gvsoc.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "pulp_nnx_ne16.h"

typedef ne16_quant_t nnx_quant_t;
typedef ne16_norm_t nnx_norm_t;
typedef ne16_task_t nnx_task_t;
typedef ne16_dev_t nnx_dev_t;
typedef ne16_pulp_conf_t nnx_bsp_conf_t;

#define nnxTaskFlagTrue ne16TaskFlagTrue
#define nnxTaskFlagFalse ne16TaskFlagFalse

#define nnx_task_init ne16_task_init
#define nnx_task_set_dims ne16_task_set_dims
#define nnx_task_set_dims_stride2x2 ne16_task_set_dims_stride2x2
#define nnx_task_set_ptrs ne16_task_set_ptrs

#define NNX_GVSOC_LOG_LEVEL NE16_GVSOC_LOG_LEVEL_ALL
#define NNX_GVSOC_LOG_FORMAT NE16_GVSOC_LOG_FORMAT_HEXADECIMAL
#define nnx_gvsoc_log_activate ne16_gvsoc_log_activate
#define nnx_gvsoc_log_deactivate ne16_gvsoc_log_deactivate

#define nnx_bsp_get_dev ne16_pulp_get_dev

#define nnx_init ne16_nnx_init
#define nnx_dispatch_wait ne16_nnx_dispatch_wait
#define nnx_dispatch_stride2x2 ne16_nnx_dispatch_stride2x2
#define nnx_dispatch ne16_nnx_dispatch
#define nnx_resolve_wait ne16_nnx_resolve_wait
#define nnx_term ne16_nnx_term

#elif defined NNX_NEUREKA

#include "neureka.h"
#include "neureka_gvsoc.h"
#include "neureka_siracusa_bsp.h"
#include "neureka_task.h"
#include "pulp_nnx_neureka.h"

typedef neureka_quant_t nnx_quant_t;
typedef neureka_norm_t nnx_norm_t;
typedef neureka_task_t nnx_task_t;
typedef neureka_dev_t nnx_dev_t;
typedef neureka_siracusa_conf_t nnx_bsp_conf_t;

#define nnxTaskFlagTrue neurekaTaskFlagTrue
#define nnxTaskFlagFalse neurekaTaskFlagFalse

#define nnx_task_init neureka_task_init
#define nnx_task_set_dims neureka_task_set_dims
#define nnx_task_set_dims_stride2x2 neureka_task_set_dims_stride2x2
#define nnx_task_set_ptrs neureka_task_set_ptrs

#define NNX_GVSOC_LOG_LEVEL NEUREKA_GVSOC_LOG_LEVEL_ALL
#define NNX_GVSOC_LOG_FORMAT NEUREKA_GVSOC_LOG_FORMAT_HEXADECIMAL
#define nnx_gvsoc_log_activate neureka_gvsoc_log_activate
#define nnx_gvsoc_log_deactivate neureka_gvsoc_log_deactivate

#define nnx_bsp_get_dev neureka_siracusa_get_dev

#define nnx_init neureka_nnx_init
#define nnx_dispatch_wait neureka_nnx_dispatch_wait
#define nnx_dispatch_stride2x2 neureka_nnx_dispatch_stride2x2
#define nnx_dispatch neureka_nnx_dispatch
#define nnx_resolve_wait neureka_nnx_resolve_wait
#define nnx_term neureka_nnx_term

#endif // NNX_NE16 || NNX_NEUREKA

// Generated headers
#include "bias.h"
#include "input.h"
#include "layer_conf.h"
#include "output.h"
#include "scale.h"
#include "weight.h"

static void task_prepare(nnx_task_t *task) {
  nnx_task_init(
      task, WEIGHT_HEIGHT, GROUPS > 1, INPUT_BITS, OUTPUT_BITS, WEIGHT_BITS,
      weightOffsetModeLayerWise, WEIGHT_OFFSET,
      (nnx_quant_t){.shift_amount = OUTSHIFT,
                    .mode = quantMode8Bit,
                    .function =
                        HAS_RELU ? quantFunctionRelu : quantFunctionIdentity,
                    .flag_rounding = nnxTaskFlagFalse},
      (nnx_norm_t){.mode = normMode8Bit,
                   .flag_bias = HAS_BIAS ? nnxTaskFlagTrue : nnxTaskFlagFalse,
                   .flag_shift = nnxTaskFlagFalse},
      STRIDE_HEIGHT);

  if (STRIDE_WIDTH == 2 && STRIDE_HEIGHT == 2) {
    nnx_task_set_dims_stride2x2(
        task, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, INPUT_WIDTH,
        INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL,
        OUTPUT_WIDTH, OUTPUT_CHANNEL, WEIGHT_HEIGHT, WEIGHT_WIDTH, PADDING_TOP,
        PADDING_BOTTOM, PADDING_RIGHT, PADDING_LEFT);
  } else {
    nnx_task_set_dims(task, INPUT_WIDTH, INPUT_CHANNEL, INPUT_WIDTH,
                      INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH,
                      OUTPUT_CHANNEL, OUTPUT_WIDTH, OUTPUT_CHANNEL, PADDING_TOP,
                      PADDING_BOTTOM, PADDING_RIGHT, PADDING_LEFT);
  }

  nnx_task_set_ptrs(task, (uint32_t)input, INPUT_WIDTH, INPUT_CHANNEL,
                    INPUT_BITS, PADDING_TOP, PADDING_LEFT, (uint32_t)output,
                    (uint32_t)weight, (uint32_t)scale, NULL,
#if HAS_BIAS == 1
                    (uint32_t)bias
#else
                    NULL
#endif
  );
}

static void task_execute(nnx_task_t *task) {
  nnx_dev_t *dev = nnx_bsp_get_dev();

  nnx_gvsoc_log_activate(dev, NNX_GVSOC_LOG_LEVEL, NNX_GVSOC_LOG_FORMAT);

  nnx_bsp_conf_t conf = {.max_stall = 8};
  nnx_init(dev, &conf);

  nnx_dispatch_wait(dev);

  if (STRIDE_WIDTH == 2 && STRIDE_HEIGHT == 2) {
    nnx_dispatch_stride2x2(dev, task, INPUT_WIDTH, INPUT_CHANNEL, INPUT_WIDTH,
                           INPUT_CHANNEL, OUTPUT_HEIGHT, OUTPUT_WIDTH,
                           OUTPUT_CHANNEL, OUTPUT_WIDTH, OUTPUT_CHANNEL,
                           WEIGHT_HEIGHT, WEIGHT_WIDTH);
  } else {
    nnx_dispatch(dev, task);
  }

  nnx_resolve_wait(dev, task);

  nnx_term(dev);

  nnx_gvsoc_log_deactivate(dev);
}

void execute_nnx_layer(void *args) {
  nnx_task_t task;
  task_prepare(&task);
  task_execute(&task);
}
