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

#include "neureka_v2_task.h"
#include "neureka_v2_task_defs.h"
#include "pulp_nnx_util.h"

uint32_t neureka_v2_get_tile_padding(uint32_t padding, uint32_t i_height,
                                     uint32_t i_width, uint32_t n_height,
                                     uint32_t n_width) {
  uint32_t tile_padding = padding;
  if (i_height > 0) {
    tile_padding &= ~(0xf << 28);
  }
  if (i_width < n_width - 1) {
    tile_padding &= ~(0xf << 24);
  }
  if (i_height < n_height - 1) {
    tile_padding &= ~(0xf << 20);
  }
  if (i_width > 0) {
    tile_padding &= ~(0xf << 16);
  }
  return tile_padding;
}

void neureka_v2_task_init(neureka_v2_task_t *task) {
  *task = (neureka_v2_task_t){.data = {0}};
}

void neureka_v2_task_set_op_to_conv(neureka_v2_task_t *task,
                                    const uint8_t kernel_shape,
                                    const uint8_t depthwise) {
  task->depthwise = depthwise;
  task->kernel_shape = kernel_shape;

  const int flag_mode = kernel_shape == 1 ? NEUREKA_V2_FLAG_MODE_1x1
                        : depthwise == 1  ? NEUREKA_V2_FLAG_MODE_3x3_DW
                                          : NEUREKA_V2_FLAG_MODE_3x3;

  task->data.cfg.conf0 &= ~(NEUREKA_V2_MASK_FLAG_MODE);
  task->data.cfg.conf0 |= flag_mode;
}

void neureka_v2_task_set_bits(neureka_v2_task_t *task, const uint8_t input_bits,
                              const uint8_t output_bits,
                              const uint8_t weight_bits) {
  neureka_v2_quant_mode_e quantMode;
  if (output_bits == 8) {
    quantMode = quantMode8Bit;
  } else {
    quantMode = quantMode32Bit;
  }

  task->qw = weight_bits;
  task->data.cfg.conf0 &=
      ~(NEUREKA_V2_MASK_QUANT_MODE | NEUREKA_V2_MASK_FLAG_WEIGHT_BITS);
  task->data.cfg.conf0 |= quantMode | (weight_bits - 1);
}

void neureka_v2_task_set_norm_quant(neureka_v2_task_t *task,
                                    neureka_v2_quant_t quant,
                                    neureka_v2_norm_t norm) {
  task->data.cfg.conf0 &=
      ~(NEUREKA_V2_MASK_QUANT_FUNCTION | NEUREKA_V2_MASK_SHIFT_AMOUNT |
        NEUREKA_V2_MASK_NORM_MODE | NEUREKA_V2_MASK_FLAG_NORM_BIAS |
        NEUREKA_V2_MASK_FLAG_NORM_SHIFT);
  task->data.cfg.conf0 |= NEUREKA_V2_FLAG_NORM_QUANT | quant.function |
                          (quant.shift_amount << 16) | norm.mode |
                          norm.flag_bias << NEUREKA_V2_SHIFT_FLAG_NORM_BIAS |
                          norm.flag_shift << NEUREKA_V2_SHIFT_FLAG_NORM_SHIFT;
}

void neureka_v2_task_set_weight_offset(neureka_v2_task_t *task,
                                       const int32_t weight_offset) {
  task->data.cfg.weight_offset_factor = weight_offset;
}

void neureka_v2_task_set_activation_signed(neureka_v2_task_t *task) {
  task->data.cfg.conf0 |= NEUREKA_V2_FLAG_ACTIVATION_SIGNED;
}

void neureka_v2_task_set_activation_unsigned(neureka_v2_task_t *task) {
  task->data.cfg.conf0 &= ~NEUREKA_V2_FLAG_ACTIVATION_SIGNED;
}

void neureka_v2_task_set_outfeat_signed(neureka_v2_task_t *task) {
  task->data.cfg.conf0 |= NEUREKA_V2_FLAG_OUTFEAT_SIGNED;
}

void neureka_v2_task_set_outfeat_unsigned(neureka_v2_task_t *task) {
  task->data.cfg.conf0 &= ~NEUREKA_V2_FLAG_OUTFEAT_SIGNED;
}

void neureka_v2_task_set_streamin_signed(neureka_v2_task_t *task) {
  task->data.cfg.conf0 |= NEUREKA_V2_FLAG_STREAMIN_SIGNED;
}

void neureka_v2_task_set_streamin_unsigned(neureka_v2_task_t *task) {
  task->data.cfg.conf0 &= ~NEUREKA_V2_FLAG_STREAMIN_SIGNED;
}

void neureka_v2_task_set_streamin(neureka_v2_task_t *task) {
  task->data.cfg.conf0 |= NEUREKA_V2_FLAG_STREAMIN;
}

void neureka_v2_task_set_infeat_prefetch(neureka_v2_task_t *task) {
  task->data.cfg.conf0 |= NEUREKA_V2_FLAG_INFEAT_PREFETCH;
}

void neureka_v2_task_set_weight_source(
    neureka_v2_task_t *task, neureka_v2_weight_source_e weight_source) {
  task->data.cfg.conf0 &= ~NEUREKA_V2_MASK_FLAG_WEIGHT_SOURCE;
  task->data.cfg.conf0 |= weight_source;
}

/** neureka_v2_pad_addr
 *
 * Calculate the pointer to the start of the ptr as if
 * it was the start to the padded data.
 * Necessary for input pointer when it's padded.
 */
uint32_t neureka_v2_pad_addr(uint32_t ptr, const uint32_t width,
                             const uint32_t width_stride,
                             const uint8_t padding_top,
                             const uint8_t padding_left) {
  return ptr - (padding_top * width + padding_left) * width_stride;
}

void neureka_v2_task_set_addr_conv(neureka_v2_task_t *task, uint32_t input_addr,
                                   uint32_t w_in, uint32_t w_in_stride,
                                   uint8_t padding_top, uint8_t padding_left,
                                   uint32_t output_addr,
                                   uint32_t weights_addr) {
  task->data.infeat_addr = neureka_v2_pad_addr(input_addr, w_in, w_in_stride,
                                               padding_top, padding_left);
  task->data.outfeat_addr = output_addr;
  if ((task->data.cfg.conf0 & NEUREKA_V2_MASK_FLAG_WEIGHT_SOURCE) ==
      NEUREKA_V2_FLAG_WEIGHT_SOURCE_WMEM) {
    // weights_addr -= 0x10400000;
  } else {
    weights_addr -= 0x10000000;
  }
  task->data.weights_addr = weights_addr;
}

void neureka_v2_task_set_addr_norm_quant(neureka_v2_task_t *task,
                                         uint32_t scale_addr,
                                         uint32_t shift_addr,
                                         uint32_t bias_addr) {
  task->data.scale_addr = scale_addr;
  task->data.scale_shift_addr = shift_addr;
  task->data.scale_bias_addr = bias_addr;
}

void neureka_v2_task_set_strides(neureka_v2_task_t *task, const uint32_t k_in,
                                 const uint32_t h_in_stride,
                                 const uint32_t w_in_stride,
                                 const uint32_t h_out_stride,
                                 const uint32_t w_out_stride) {
  const uint32_t num_k_in = nnx_calculate_number_of_tiles(
      k_in, NEUREKA_V2_SUBTILE_INPUT_OUTPUT_CHANNEL);

  const neureka_v2_stride_t input_stride = {
      .d0 = w_in_stride, .d1 = h_in_stride, .d2 = 0};
  task->data.cfg.input_stride = input_stride;

  const neureka_v2_stride_t output_stride = {
      .d0 = NEUREKA_V2_OUTPUT_BANDWIDTH_BYTES,
      .d1 = w_out_stride,
      .d2 = h_out_stride};
  task->data.cfg.output_stride = output_stride;

  task->data.cfg.weights_stride.d0 = NEUREKA_V2_WEIGHT_BANDWIDTH_BYTES;
  if (task->kernel_shape == 1) { // 1x1
    task->data.cfg.weights_stride.d1 =
        num_k_in * task->qw * NEUREKA_V2_SUBTILE_INPUT_OUTPUT_CHANNEL / 8;
  } else if (!task->depthwise) { // 3x3
    task->data.cfg.weights_stride.d1 =
        NEUREKA_V2_WEIGHT_BANDWIDTH_BYTES * task->qw * num_k_in;
  } else { // 3x3 depthwise
    task->data.cfg.weights_stride.d1 = 0;
  }
  task->data.cfg.weights_stride.d2 = 0;
}

void neureka_v2_task_set_counters(neureka_v2_task_t *task, const uint32_t k_in,
                                  const uint32_t h_out, const uint32_t w_out,
                                  const uint32_t k_out,
                                  const uint8_t padding_bottom,
                                  const uint8_t padding_right) {
  const uint16_t num_Ko = nnx_calculate_number_of_tiles(
      k_out, NEUREKA_V2_SUBTILE_INPUT_OUTPUT_CHANNEL);
  const uint16_t num_Ki = nnx_calculate_number_of_tiles(
      k_in, NEUREKA_V2_SUBTILE_INPUT_OUTPUT_CHANNEL);
  const uint16_t num_Ho =
      nnx_calculate_number_of_tiles(h_out, NEUREKA_V2_SUBTILE_OUTPUT_HEIGHT);
  const uint16_t num_Wo =
      nnx_calculate_number_of_tiles(w_out, NEUREKA_V2_SUBTILE_OUTPUT_WIDTH);

  const uint16_t rem_Ko = nnx_calculate_last_tile_size(
      k_out, NEUREKA_V2_SUBTILE_INPUT_OUTPUT_CHANNEL);
  const uint16_t rem_Ki = nnx_calculate_last_tile_size(
      k_in, NEUREKA_V2_SUBTILE_INPUT_OUTPUT_CHANNEL);
  const uint16_t rem_Ho =
      nnx_calculate_last_tile_size(h_out, NEUREKA_V2_SUBTILE_OUTPUT_HEIGHT);
  const uint16_t rem_Wo =
      nnx_calculate_last_tile_size(w_out, NEUREKA_V2_SUBTILE_OUTPUT_WIDTH);
  const uint16_t rem_Hi = (task->kernel_shape == 1 ? rem_Ho : rem_Ho + 2);
  const uint16_t rem_Wi = (task->kernel_shape == 1 ? rem_Wo : rem_Wo + 2);

  const neureka_v2_subtile_t subtile = {
      .number = {.KoKi = nnx_concat_half(num_Ko, num_Ki),
                 .HoWo = nnx_concat_half(num_Ho, num_Wo)},
      .remainder = {.KoKi = nnx_concat_half(rem_Ko, rem_Ki),
                    .HoWo = nnx_concat_half(rem_Ho, rem_Wo),
                    .HiWi = nnx_concat_half(rem_Hi, rem_Wi)}};
  task->data.cfg.subtile = subtile;
}

void neureka_v2_task_set_padding(neureka_v2_task_t *task, const uint8_t top,
                                 const uint8_t bottom, const uint8_t left,
                                 const uint8_t right, const uint8_t value) {
  task->data.cfg.padding = ((top & 0xf) << 28) | ((right & 0xf) << 24) |
                           ((bottom & 0xf) << 20) | ((left & 0xf) << 16) |
                           (value & 0xff);
}

void neureka_v2_task_set_mask_filter(neureka_v2_task_t *task, const uint8_t top,
                                     const uint8_t bottom, const uint8_t left,
                                     const uint8_t right) {
  task->data.cfg.filter_mask = ((top & 0xff) << 24) | ((right & 0xff) << 16) |
                               ((bottom & 0xff) << 8) | ((left & 0xff) << 0);
}

void neureka_v2_task_set_dims(
    neureka_v2_task_t *task, const uint32_t w_in, const uint32_t k_in,
    const uint32_t h_in_stride, const uint32_t w_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t h_out_stride, const uint32_t w_out_stride,
    const uint8_t padding_top, const uint8_t padding_bottom,
    const uint8_t padding_left, const uint8_t padding_right) {
  neureka_v2_task_set_strides(task, k_in, h_in_stride, w_in_stride,
                              h_out_stride, w_out_stride);
  neureka_v2_task_set_counters(task, k_in, h_out, w_out, k_out, padding_bottom,
                               padding_right);
  neureka_v2_task_set_padding(task, padding_top, padding_bottom, padding_left,
                              padding_right, 0);
}
