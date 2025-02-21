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

typedef ne16_norm_mode_e nnx_norm_mode_e;
typedef ne16_quant_t nnx_quant_t;
typedef ne16_quant_function_e nnx_quant_function_e;
typedef ne16_norm_t nnx_norm_t;
typedef ne16_task_t nnx_task_t;
typedef ne16_dev_t nnx_dev_t;
typedef ne16_pulp_conf_t nnx_bsp_conf_t;
typedef ne16_task_flag_e nnx_task_flag_e;

#define nnxTaskFlagTrue ne16TaskFlagTrue
#define nnxTaskFlagFalse ne16TaskFlagFalse

#define nnx_task_init ne16_task_init
#define nnx_task_set_op_to_conv ne16_task_set_op_to_conv
#define nnx_task_set_bits ne16_task_set_bits
#define nnx_task_set_norm_quant ne16_task_set_norm_quant
#define nnx_task_set_weight_offset ne16_task_set_weight_offset
#define nnx_task_set_dims ne16_task_set_dims
#define nnx_task_set_dims_stride2x2 ne16_task_set_dims_stride2x2
#define nnx_task_set_addr_conv ne16_task_set_addr_conv
#define nnx_task_set_addr_norm_quant ne16_task_set_addr_norm_quant

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

typedef neureka_norm_mode_e nnx_norm_mode_e;
typedef neureka_quant_t nnx_quant_t;
typedef neureka_quant_function_e nnx_quant_function_e;
typedef neureka_norm_t nnx_norm_t;
typedef neureka_task_t nnx_task_t;
typedef neureka_dev_t nnx_dev_t;
typedef neureka_siracusa_conf_t nnx_bsp_conf_t;
typedef neureka_task_flag_e nnx_task_flag_e;

#define nnxTaskFlagTrue neurekaTaskFlagTrue
#define nnxTaskFlagFalse neurekaTaskFlagFalse

#define nnx_task_init neureka_task_init
#define nnx_task_set_op_to_conv neureka_task_set_op_to_conv
#define nnx_task_set_bits neureka_task_set_bits
#define nnx_task_set_norm_quant neureka_task_set_norm_quant
#define nnx_task_set_weight_offset neureka_task_set_weight_offset
#define nnx_task_set_dims neureka_task_set_dims
#define nnx_task_set_addr_conv neureka_task_set_addr_conv
#define nnx_task_set_addr_norm_quant neureka_task_set_addr_norm_quant

#define NNX_GVSOC_LOG_LEVEL NEUREKA_GVSOC_LOG_LEVEL_ALL
#define NNX_GVSOC_LOG_FORMAT NEUREKA_GVSOC_LOG_FORMAT_HEXADECIMAL
#define nnx_gvsoc_log_activate neureka_gvsoc_log_activate
#define nnx_gvsoc_log_deactivate neureka_gvsoc_log_deactivate

#define nnx_bsp_get_dev neureka_siracusa_get_dev

#define nnx_init neureka_nnx_init
#define nnx_dispatch_wait neureka_nnx_dispatch_wait
#define nnx_dispatch neureka_nnx_dispatch
#define nnx_resolve_wait neureka_nnx_resolve_wait
#define nnx_term neureka_nnx_term

#elif defined NNX_NEUREKA_V2

#include "neureka_v2.h"
#include "neureka_v2_gvsoc.h"
#include "neureka_v2_siracusa_bsp.h"
#include "neureka_v2_task.h"
#include "pulp_nnx_neureka_v2.h"

typedef neureka_v2_norm_mode_e nnx_norm_mode_e;
typedef neureka_v2_quant_t nnx_quant_t;
typedef neureka_v2_quant_function_e nnx_quant_function_e;
typedef neureka_v2_norm_t nnx_norm_t;
typedef neureka_v2_task_t nnx_task_t;
typedef neureka_v2_dev_t nnx_dev_t;
typedef neureka_v2_siracusa_conf_t nnx_bsp_conf_t;
typedef neureka_v2_task_flag_e nnx_task_flag_e;

#define nnxTaskFlagTrue neurekaV2TaskFlagTrue
#define nnxTaskFlagFalse neurekaV2TaskFlagFalse

#define nnx_task_init neureka_v2_task_init
#define nnx_task_set_op_to_conv neureka_v2_task_set_op_to_conv
#define nnx_task_set_bits neureka_v2_task_set_bits
#define nnx_task_set_norm_quant neureka_v2_task_set_norm_quant
#define nnx_task_set_weight_offset neureka_v2_task_set_weight_offset
#define nnx_task_set_dims neureka_v2_task_set_dims
#define nnx_task_set_addr_conv neureka_v2_task_set_addr_conv
#define nnx_task_set_addr_norm_quant neureka_v2_task_set_addr_norm_quant

#define NNX_GVSOC_LOG_LEVEL NEUREKA_V2_GVSOC_LOG_LEVEL_ALL
#define NNX_GVSOC_LOG_FORMAT NEUREKA_V2_GVSOC_LOG_FORMAT_HEXADECIMAL
#define nnx_gvsoc_log_activate neureka_v2_gvsoc_log_activate
#define nnx_gvsoc_log_deactivate neureka_v2_gvsoc_log_deactivate

#define nnx_bsp_get_dev neureka_v2_siracusa_get_dev

#define nnx_init neureka_v2_nnx_init
#define nnx_dispatch_wait neureka_v2_nnx_dispatch_wait
#define nnx_dispatch neureka_v2_nnx_dispatch
#define nnx_resolve_wait neureka_v2_nnx_resolve_wait
#define nnx_term neureka_v2_nnx_term

#endif // NNX_NE16 || NNX_NEUREKA || NNX_NEUREKA_V2

// Generated headers
#include "input.h"
#include "output.h"
#include "weight.h"

#include "layer_conf.h"

// The HAS_NORM_QUANT and HAS_BIAS are defined in layer_conf.h
#if HAS_NORM_QUANT != 0
#include "scale.h"
#if HAS_BIAS != 0
#include "bias.h"
#endif
#endif

static void task_prepare(nnx_task_t *task) {
  nnx_task_init(task);
#if defined NNX_NEUREKA || defined NNX_NEUREKA_V2
  nnx_task_set_op_to_conv(task, WEIGHT_HEIGHT, GROUPS > 1);
#else
  nnx_task_set_op_to_conv(task, WEIGHT_HEIGHT, GROUPS > 1, STRIDE_HEIGHT);
#endif
  nnx_task_set_bits(task, INPUT_BITS, OUTPUT_BITS, WEIGHT_BITS);

#if defined NNX_NE16 || defined NNX_NEUREKA
  nnx_task_set_weight_offset(task, weightOffsetModeLayerWise, WEIGHT_OFFSET);
#elif defined NNX_NEUREKA_V2
  nnx_task_set_weight_offset(task, WEIGHT_OFFSET);
#endif

#ifdef NNX_NEUREKA
#if INPUT_SIGNED == 1
  neureka_task_set_input_signed(task);
#else
  neureka_task_set_input_unsigned(task);
#endif
#if defined WMEM_SRAM || defined WMEM_MRAM
  neureka_task_set_weight_source(task, neurekaWeightSourceWmem);
#else
  neureka_task_set_weight_source(task, neurekaWeightSourceTcdm);
#endif
#endif

#ifdef NNX_NEUREKA_V2
#if INPUT_SIGNED == 1
  neureka_v2_task_set_activation_signed(task);
#else
  neureka_v2_task_set_activation_unsigned(task);
#endif
#if OUTPUT_SIGNED == 1
  neureka_v2_task_set_outfeat_signed(task);
#else
  neureka_v2_task_set_outfeat_unsigned(task);
#endif
#if defined WMEM_SRAM || defined WMEM_MRAM
  neureka_v2_task_set_weight_source(task, neurekaV2WeightSourceWmem);
#else
  neureka_v2_task_set_weight_source(task, neurekaV2WeightSourceTcdm);
#endif
#endif

  const uint32_t w_in_stride = INPUT_CHANNEL * INPUT_BITS / 8;
  const uint32_t h_in_stride = INPUT_WIDTH * w_in_stride;
  const uint32_t w_out_stride = OUTPUT_CHANNEL * OUTPUT_BITS / 8;
  const uint32_t h_out_stride = OUTPUT_WIDTH * w_out_stride;

#if STRIDE_HEIGHT == 2 && STRIDE_WIDTH == 2
  nnx_task_set_dims_stride2x2(
      task, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL, h_in_stride, w_in_stride,
      OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL, h_out_stride, w_out_stride,
      WEIGHT_HEIGHT, WEIGHT_WIDTH, PADDING_TOP, PADDING_BOTTOM, PADDING_LEFT,
      PADDING_RIGHT);
#else
  nnx_task_set_dims(task, INPUT_WIDTH, INPUT_CHANNEL, h_in_stride, w_in_stride,
                    OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_CHANNEL, h_out_stride,
                    w_out_stride, PADDING_TOP, PADDING_BOTTOM, PADDING_LEFT,
                    PADDING_RIGHT);
#endif

  nnx_task_set_addr_conv(task, (uint32_t)input, INPUT_WIDTH, w_in_stride,
                         PADDING_TOP, PADDING_LEFT, (uint32_t)output,
                         (uint32_t)weight);

#if HAS_NORM_QUANT == 1
#if SCALE_BITS == 8
  const nnx_norm_mode_e normMode = normMode8Bit;
#elif SCALE_BITS == 32
  const nnx_norm_mode_e normMode = normMode32Bit;
#endif

  const nnx_task_flag_e flag_bias =
      HAS_BIAS ? nnxTaskFlagTrue : nnxTaskFlagFalse;
#if HAS_BIAS == 1
  const uint32_t bias_addr = (uint32_t)bias;
#else
  const uint32_t bias_addr = (uint32_t)NULL;
#endif

  nnx_quant_function_e quant_function =
      HAS_RELU ? quantFunctionRelu : quantFunctionIdentity;

  nnx_task_set_norm_quant(task,
                          (nnx_quant_t){.shift_amount = OUTSHIFT,
                                        .function = quant_function,
                                        .flag_rounding = nnxTaskFlagFalse},
                          (nnx_norm_t){.mode = normMode,
                                       .flag_bias = flag_bias,
                                       .flag_shift = nnxTaskFlagFalse});

  nnx_task_set_addr_norm_quant(task, (uint32_t)scale, (uint32_t)NULL,
                               bias_addr);
#endif // HAS_NORM_QUANT
}

static void task_execute(nnx_task_t *task) {
  const nnx_dev_t *dev = nnx_bsp_get_dev();

#if __PLATFORM__ == ARCHI_PLATFORM_GVSOC
  nnx_gvsoc_log_activate(dev, NNX_GVSOC_LOG_LEVEL, NNX_GVSOC_LOG_FORMAT);
#endif

  nnx_bsp_conf_t conf = {.max_stall = 8};
  nnx_init(dev, &conf);

  nnx_dispatch_wait(dev);

#if STRIDE_HEIGHT == 2 && STRIDE_WIDTH == 2
  nnx_dispatch_stride2x2(dev, task, INPUT_WIDTH, INPUT_CHANNEL, OUTPUT_HEIGHT,
                         OUTPUT_WIDTH, OUTPUT_CHANNEL, WEIGHT_HEIGHT,
                         WEIGHT_WIDTH);
#else
  nnx_dispatch(dev, task);
#endif

  nnx_resolve_wait(dev, task);

  nnx_term(dev);

#if __PLATFORM__ == ARCHI_PLATFORM_GVSOC
  nnx_gvsoc_log_deactivate(dev);
#endif
}

void execute_nnx_layer(void *args) {
  nnx_task_t task;
  task_prepare(&task);
  task_execute(&task);
}
