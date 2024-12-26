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

#ifndef __NEUREKA_V2_TASK_H__
#define __NEUREKA_V2_TASK_H__

#include "neureka_v2_task_defs.h"
#include <stdint.h>

typedef enum neureka_v2_task_flag_e {
  neurekaV2TaskFlagFalse = 0,
  neurekaV2TaskFlagTrue = 1
} neureka_v2_task_flag_e;

typedef enum neureka_v2_weight_source_e {
  neurekaV2WeightSourceTcdm = NEUREKA_V2_FLAG_WEIGHT_SOURCE_TCDM,
  neurekaV2WeightSourceWmem = NEUREKA_V2_FLAG_WEIGHT_SOURCE_WMEM
} neureka_v2_weight_source_e;

typedef enum {
  normMode8Bit = NEUREKA_V2_NORM_MODE_8BIT,
  normMode32Bit = NEUREKA_V2_NORM_MODE_32BIT
} neureka_v2_norm_mode_e;

typedef struct neureka_v2_norm_t {
  neureka_v2_norm_mode_e mode;
  neureka_v2_task_flag_e flag_bias;
  neureka_v2_task_flag_e flag_shift;
} neureka_v2_norm_t;

typedef enum neureka_v2_quant_mode_e {
  quantMode8Bit = NEUREKA_V2_QUANT_MODE_8BIT,
  quantMode32Bit = NEUREKA_V2_QUANT_MODE_32BIT
} neureka_v2_quant_mode_e;

typedef enum neureka_v2_quant_function_e {
  quantFunctionIdentity = NEUREKA_V2_FLAG_QUANT_FUNCTION_IDENTITY,
  quantFunctionRelu = NEUREKA_V2_FLAG_QUANT_FUNCTION_RELU
} neureka_v2_quant_function_e;

typedef struct neureka_v2_quant_t {
  // Shift amount must be in range 0x00-0x1F
  uint8_t shift_amount;
  neureka_v2_quant_function_e function;
  neureka_v2_task_flag_e flag_rounding;
} neureka_v2_quant_t;

typedef struct neureka_v2_stride_t {
  uint32_t d0;
  uint32_t d1;
  uint32_t d2;
} neureka_v2_stride_t;

typedef struct neureka_v2_subtile_remainder_t {
  uint32_t KoKi;
  uint32_t HoWo;
  uint32_t HiWi;
} neureka_v2_subtile_remainder_t;

typedef struct neureka_v2_subtile_number_t {
  uint32_t KoKi;
  uint32_t HoWo;
} neureka_v2_subtile_number_t;

typedef struct neureka_v2_subtile_t {
  neureka_v2_subtile_remainder_t remainder;
  neureka_v2_subtile_number_t number;
} neureka_v2_subtile_t;

typedef struct neureka_v2_cfg_t {
  neureka_v2_stride_t input_stride;
  neureka_v2_stride_t output_stride;
  neureka_v2_stride_t weights_stride;
  neureka_v2_subtile_t subtile;
  uint32_t padding;
  uint32_t weight_offset_factor;
  uint32_t filter_mask;
  uint32_t conf0;
} neureka_v2_cfg_t;

typedef struct neureka_v2_task_data_t {
  uint32_t weights_addr;
  uint32_t infeat_addr;
  uint32_t outfeat_addr;
  uint32_t scale_addr;
  uint32_t scale_shift_addr;
  uint32_t scale_bias_addr;
  neureka_v2_cfg_t cfg;
  uint32_t streamin_addr;
} neureka_v2_task_data_t;

typedef struct neureka_v2_task_t {
  neureka_v2_task_data_t data;
  uint8_t qw;
  uint8_t subtile_output_channel;
  uint8_t subtile_input_channel;
  uint8_t kernel_shape;
  uint8_t depthwise;
  uint8_t id;
} neureka_v2_task_t;

void neureka_v2_task_init(neureka_v2_task_t *task);
void neureka_v2_task_set_op_to_conv(neureka_v2_task_t *task,
                                    const uint8_t kernel_shape,
                                    const uint8_t depthwise);
void neureka_v2_task_set_bits(neureka_v2_task_t *task, const uint8_t input_bits,
                              const uint8_t output_bits,
                              const uint8_t weight_bits);
void neureka_v2_task_set_norm_quant(neureka_v2_task_t *task,
                                    neureka_v2_quant_t quant,
                                    neureka_v2_norm_t norm);
void neureka_v2_task_set_weight_offset(neureka_v2_task_t *task,
                                       const int32_t weight_offset);
void neureka_v2_task_set_activation_signed(neureka_v2_task_t *task);
void neureka_v2_task_set_activation_unsigned(neureka_v2_task_t *task);
void neureka_v2_task_set_outfeat_signed(neureka_v2_task_t *task);
void neureka_v2_task_set_outfeat_unsigned(neureka_v2_task_t *task);
void neureka_v2_task_set_streamin_signed(neureka_v2_task_t *task);
void neureka_v2_task_set_streamin_unsigned(neureka_v2_task_t *task);
void neureka_v2_task_set_streamin(neureka_v2_task_t *task);
void neureka_v2_task_set_weight_source(
    neureka_v2_task_t *task, neureka_v2_weight_source_e weight_source);
uint32_t neureka_v2_get_tile_padding(uint32_t padding, uint32_t i_height,
                                     uint32_t i_width, uint32_t n_height,
                                     uint32_t n_width);
uint32_t neureka_v2_pad_addr(uint32_t ptr, const uint32_t width,
                             const uint32_t width_stride,
                             const uint8_t padding_top,
                             const uint8_t padding_left);
void neureka_v2_task_set_addr_conv(neureka_v2_task_t *task, uint32_t input_addr,
                                   uint32_t w_in, uint32_t w_in_stride,
                                   uint8_t padding_top, uint8_t padding_left,
                                   uint32_t output_addr, uint32_t weights_addr);
void neureka_v2_task_set_addr_norm_quant(neureka_v2_task_t *task,
                                         uint32_t scale_addr,
                                         uint32_t shift_addr,
                                         uint32_t bias_addr);
/** neureka_v2_task_set_strides
 *
 * All the strides variables are strides between elements alongside that
 * dimension and expressed in bytes. There is no stride variable for the channel
 * dimension because the N-EUREKA requires the channels to be contiguous.
 */
void neureka_v2_task_set_strides(neureka_v2_task_t *task, const uint32_t k_in,
                                 const uint32_t h_in_stride,
                                 const uint32_t w_in_stride,
                                 const uint32_t h_out_stride,
                                 const uint32_t w_out_stride);
void neureka_v2_task_set_counters(neureka_v2_task_t *task, const uint32_t k_in,
                                  const uint32_t h_out, const uint32_t w_out,
                                  const uint32_t k_out,
                                  const uint8_t padding_bottom,
                                  const uint8_t padding_right);
void neureka_v2_task_set_padding(neureka_v2_task_t *task, const uint8_t top,
                                 const uint8_t bottom, const uint8_t left,
                                 const uint8_t right, const uint8_t value);
void neureka_v2_task_set_mask_filter(neureka_v2_task_t *task, const uint8_t top,
                                     const uint8_t bottom, const uint8_t left,
                                     const uint8_t right);
/** neureka_v2_task_set_dims
 *
 * All the strides variables are strides between elements alongside that
 * dimension and expressed in bytes. There is no stride variable for the channel
 * dimension because the N-EUREKA requires the channels to be contiguous.
 */
void neureka_v2_task_set_dims(
    neureka_v2_task_t *task, const uint32_t w_in, const uint32_t k_in,
    const uint32_t h_in_stride, const uint32_t w_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t h_out_stride, const uint32_t w_out_stride,
    const uint8_t padding_top, const uint8_t padding_bottom,
    const uint8_t padding_left, const uint8_t padding_right);

#endif // !__NEUREKA_V2_TASK_H__
