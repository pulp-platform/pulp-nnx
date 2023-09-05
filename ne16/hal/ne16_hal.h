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

#ifndef __NE16_HAL_H__
#define __NE16_HAL_H__

#include "ne16_defs.h"
#include <stdint.h>

#define NE16_WRITE(offset, value)                                              \
  *(int volatile *)(NE16_BASE_ADDR + (offset)) = (value)
#define NE16_READ(offset) *(int volatile *)(NE16_BASE_ADDR + (offset))

#define NE16_WRITE_IO_REG(offset, value)                                       \
  NE16_WRITE(NE16_REGISTER_OFFSET + (offset), (value))
#define NE16_READ_IO_REG(offset) NE16_READ(NE16_REGISTER_OFFSET + (offset))

#define NE16_FLAG_USED (1)
#define NE16_FLAG_UNUSED (0)

typedef enum nnx_weight_offset_mode_e {
  weightOffsetModeSymmetric = NE16_FLAG_WEIGHT_OFFSET_SYMMETRIC,
  weightOffsetModeLayerWise = NE16_FLAG_WEIGHT_OFFSET_LAYER_WISE
} nnx_weight_offset_mode_e;

typedef enum {
  normMode8Bit = NE16_NORM_MODE_8BIT,
  normMode16Bit = NE16_NORM_MODE_16BIT,
  normMode32Bit = NE16_NORM_MODE_32BIT
} nnx_norm_mode_e;

typedef struct nnx_norm_t {
  nnx_norm_mode_e mode;
  int flag_bias;
  int flag_shift;
} nnx_norm_t;

typedef enum nnx_quant_mode_e {
  quantMode8Bit = NE16_QUANT_MODE_8BIT,
  quantMode16Bit = NE16_QUANT_MODE_16BIT,
  quantMode32Bit = NE16_QUANT_MODE_32BIT
} nnx_quant_mode_e;

typedef enum nnx_quant_function_e {
  quantFunctionIdentity = NE16_FLAG_QUANT_FUNCTION_IDENTITY,
  quantFunctionRelu = NE16_FLAG_QUANT_FUNCTION_RELU
} nnx_quant_function_e;

typedef struct nnx_quant_t {
  // Shift amount must be in range 0x00-0x1F
  unsigned shift_amount;
  nnx_quant_mode_e mode;
  nnx_quant_function_e function;
  int flag_rounding;
} nnx_quant_t;

typedef struct nnx_stride_t {
  uint32_t d0;
  uint32_t d1;
  uint32_t d2;
} nnx_stride_t;

typedef struct nnx_subtile_remainder_t {
  uint32_t KoKi;
  uint32_t HoWo;
  uint32_t HiWi;
} nnx_subtile_remainder_t;

typedef struct nnx_subtile_number_t {
  uint32_t KoKi;
  uint32_t HoWo;
} nnx_subtile_number_t;

typedef struct nnx_subtile_t {
  nnx_subtile_remainder_t remainder;
  nnx_subtile_number_t number;
} nnx_subtile_t;

typedef struct nnx_cfg_t {
  nnx_stride_t input_stride;
  nnx_stride_t output_stride;
  nnx_stride_t weights_stride;
  nnx_subtile_t subtile;
  uint32_t padding;
  uint32_t weight_offset_factor;
  uint32_t filter_mask;
  uint32_t conf0;
} nnx_cfg_t;

typedef struct nnx_task_data_t {
  uint32_t weights_ptr;
  uint32_t infeat_ptr;
  uint32_t outfeat_ptr;
  uint32_t scale_ptr;
  uint32_t scale_shift_ptr;
  uint32_t scale_bias_ptr;
  nnx_cfg_t cfg;
} nnx_task_data_t;

typedef struct nnx_task_t {
  nnx_task_data_t data;
  uint8_t outbytes;
  uint8_t weight_d0_stride;
  uint8_t qw;
  uint8_t stride_shift;
  uint8_t output_channel_throughput;
  uint8_t kernel_shape;
  uint8_t depthwise;
  uint8_t id;
} nnx_task_t;

void ne16_cg_enable();
void ne16_cg_disable();

/**
 * ne16_setpriority_ne16
 *
 * Set HCI interconnect bus priority to prioritize NE16.
 */
void ne16_setpriority_ne16();

/**
 * ne16_setpriority_core
 *
 * Set HCI bus priority to prioritize cores.
 */
void ne16_setpriority_core();

/**
 * ne16_reset_maxstall
 *
 * Reset the HCI bus maxstall parameter.
 * TODO: Check if it disables it also or just resets?
 */
void ne16_reset_max_stall();

/**
 * ne16_set_maxstall
 *
 * Set the HCI bus maxstall. Maxstall defines how many cycles
 * will the HCI bus stall the lower priority master, i.e. ne16 or core,
 * before letting it do a transaction.
 */
void ne16_set_max_stall(uint32_t max_stall);
void ne16_soft_clear();
int ne16_empty();
int ne16_full();
uint8_t ne16_last_task_id();
void ne16_event_wait();
uint8_t ne16_acquire();
void ne16_run_async();
void ne16_commit();
uint32_t ne16_get_tile_padding(uint32_t padding, uint32_t i_height,
                               uint32_t i_width, uint32_t n_height,
                               uint32_t n_width);

void ne16_task_init(nnx_task_t *task, const uint8_t kernel_shape,
                    const uint8_t depthwise, const uint8_t input_bits,
                    const uint8_t output_bits, const uint8_t weights_bits,
                    const nnx_weight_offset_mode_e weights_offset_mode,
                    const uint32_t weights_offset_factor, nnx_quant_t quant,
                    nnx_norm_t norm, const uint8_t stride);
void ne16_task_set_strides(nnx_task_t *task, const uint32_t k_in,
                           const uint32_t w_in_stride,
                           const uint32_t k_in_stride,
                           const uint32_t w_out_stride,
                           const uint32_t k_out_stride);
void ne16_task_set_counters(nnx_task_t *task, const uint32_t k_in,
                            const uint32_t h_out, const uint32_t w_out,
                            const uint32_t k_out, const uint8_t padding_bottom,
                            const uint8_t padding_right);
void ne16_task_set_padding(nnx_task_t *task, const uint8_t top,
                           const uint8_t bottom, const uint8_t left,
                           const uint8_t right, const uint8_t value);
void ne16_task_set_mask_filter(nnx_task_t *task, const uint8_t top,
                               const uint8_t right, const uint8_t bottom,
                               const uint8_t left);
void ne16_task_offload(nnx_task_t *task);

#endif // __NE16_HAL_H__
