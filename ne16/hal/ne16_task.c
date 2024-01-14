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

#include "ne16_task.h"
#include "ne16_task_defs.h"
#include "pulp_nnx_util.h"

inline uint32_t ne16_get_tile_padding(uint32_t padding, uint32_t i_height,
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

void ne16_task_init(ne16_task_t *task, const uint8_t kernel_shape,
                    const uint8_t depthwise, const uint8_t input_bits,
                    const uint8_t output_bits, const uint8_t weights_bits,
                    const ne16_weight_offset_mode_e weights_offset_mode,
                    const uint32_t weights_offset_factor, ne16_quant_t quant,
                    ne16_norm_t norm, const uint8_t stride) {
  const uint32_t flag_mode16 =
      input_bits == 16 ? NE16_FLAG_MODE16 : NE16_FLAG_MODE_BASIC;

  *task = (ne16_task_t){
      .outbytes = output_bits / 8,
      .weight_d0_stride = flag_mode16 ? NE16_WEIGHT_D0_STRIDE_MODE16
                                      : NE16_WEIGHT_D0_STRIDE_MODE8,
      .qw = weights_bits,
      .stride_shift = stride == 2 ? 1 : 0,
      .output_channel_throughput = depthwise ? NE16_INPUT_CHANNEL_THROUGHPUT
                                             : NE16_OUTPUT_CHANNEL_THROUGHPUT,
      .kernel_shape = kernel_shape,
      .depthwise = depthwise,
      .data = {0}};

  const int flag_stride2x2 = stride == 2 ? NE16_FLAG_STRIDE_2x2 : 0;

  const int flag_mode = kernel_shape == 1 ? NE16_FLAG_MODE_1x1
                        : depthwise == 1  ? NE16_FLAG_MODE_3x3_DW
                                          : NE16_FLAG_MODE_3x3;

  task->data.cfg.conf0 |=
      NE16_FLAG_NORM_QUANT | quant.function | quant.mode |
      (quant.shift_amount << 16) | quant.flag_rounding << NE16_SHIFT_ROUNDING |
      norm.mode | norm.flag_bias << NE16_SHIFT_FLAG_NORM_BIAS |
      norm.flag_shift << NE16_SHIFT_FLAG_NORM_SHIFT | weights_offset_mode |
      flag_mode | flag_mode16 | (weights_bits - 1) | flag_stride2x2;

  task->data.cfg.weight_offset_factor = weights_offset_factor;
}

/** ne16_pad_ptr
 *
 * Calculate the pointer to the start of the ptr as if
 * it was the start to the padded data.
 * Necessary for input pointer when it's padded.
 */
inline uint32_t ne16_pad_ptr(uint32_t ptr, const uint32_t width,
                             const uint32_t channel, const uint8_t bits,
                             const uint8_t padding_top,
                             const uint8_t padding_left) {
  return ptr - (padding_top * width + padding_left) * channel * bits / 8;
}

inline void ne16_task_set_ptrs(ne16_task_t *task, uint32_t input_ptr,
                               uint32_t w_in, uint32_t k_in, uint8_t bits_in,
                               uint8_t padding_top, uint8_t padding_left,
                               uint32_t output_ptr, uint32_t weights_ptr,
                               uint32_t scale_ptr, uint32_t shift_ptr,
                               uint32_t bias_ptr) {
  task->data.infeat_ptr =
      ne16_pad_ptr(input_ptr, w_in, k_in, bits_in, padding_top, padding_left);
  task->data.outfeat_ptr = output_ptr;
  task->data.weights_ptr = weights_ptr;
  task->data.scale_ptr = scale_ptr;
  task->data.scale_shift_ptr = shift_ptr;
  task->data.scale_bias_ptr = bias_ptr;
}

void ne16_task_set_strides(ne16_task_t *task, const uint32_t k_in,
                           const uint32_t w_in_stride,
                           const uint32_t k_in_stride,
                           const uint32_t w_out_stride,
                           const uint32_t k_out_stride) {
  const uint32_t num_k_in = divnceil(k_in, NE16_INPUT_CHANNEL_THROUGHPUT);

  const ne16_stride_t input_stride = {
      .d0 = k_in_stride,
      .d1 = k_in_stride * w_in_stride,
      .d2 = task->depthwise ? 0
                            : k_in_stride * NE16_FILTER_BUFFER_SIZE *
                                  NE16_FILTER_BUFFER_SIZE};
  task->data.cfg.input_stride = input_stride;

  // WARNING: Stride works only for even output channel sizes (divisible by 2)
  const ne16_stride_t output_stride = {
      .d0 = 32,
      .d1 = (k_out_stride * task->outbytes) >> task->stride_shift,
      .d2 =
          (k_out_stride * task->outbytes * w_out_stride) >> task->stride_shift};
  task->data.cfg.output_stride = output_stride;

  if (task->kernel_shape == 1) {
    task->data.cfg.weights_stride.d0 = task->weight_d0_stride * task->qw;
    task->data.cfg.weights_stride.d1 =
        task->weight_d0_stride * task->qw * num_k_in;
    task->data.cfg.weights_stride.d2 = 0;
  } else if (!task->depthwise) {
    task->data.cfg.weights_stride.d0 =
        NE16_FILTER_SIZE * NE16_FILTER_SIZE * task->weight_d0_stride;
    task->data.cfg.weights_stride.d1 = NE16_FILTER_SIZE * NE16_FILTER_SIZE *
                                       task->weight_d0_stride * task->qw *
                                       num_k_in;
    task->data.cfg.weights_stride.d2 = 0;
  } else {
    task->data.cfg.weights_stride.d0 =
        NE16_FILTER_SIZE * NE16_FILTER_SIZE * task->weight_d0_stride;
    task->data.cfg.weights_stride.d1 = 0;
    task->data.cfg.weights_stride.d2 = 0;
  }
}

void ne16_task_set_counters(ne16_task_t *task, const uint32_t k_in,
                            const uint32_t h_out, const uint32_t w_out,
                            const uint32_t k_out, const uint8_t padding_bottom,
                            const uint8_t padding_right) {
  const uint16_t num_Ko = divnceil(k_out, task->output_channel_throughput);
  const uint16_t num_Ki = divnceil(k_in, NE16_INPUT_CHANNEL_THROUGHPUT);
  const uint16_t num_Ho = divnceil(h_out, NE16_FILTER_SIZE);
  const uint16_t num_Wo = divnceil(w_out, NE16_FILTER_SIZE);

  const uint16_t rem_Ko = remainder(k_out, task->output_channel_throughput);
  const uint16_t rem_Ki = remainder(k_in, NE16_INPUT_CHANNEL_THROUGHPUT);
  const uint16_t rem_Ho = remainder(h_out, NE16_FILTER_SIZE);
  const uint16_t rem_Wo = remainder(w_out, NE16_FILTER_SIZE);
  const uint16_t rem_Hi =
      (task->kernel_shape == 1 ? rem_Ho : rem_Ho + 2) - padding_bottom;
  const uint16_t rem_Wi =
      (task->kernel_shape == 1 ? rem_Wo : rem_Wo + 2) - padding_right;

  const ne16_subtile_t subtile = {
      .number = {.KoKi = concat_half(num_Ko, num_Ki),
                 .HoWo = concat_half(num_Ho, num_Wo)},
      .remainder = {.KoKi = concat_half(rem_Ko, rem_Ki),
                    .HoWo = concat_half(rem_Ho, rem_Wo),
                    .HiWi = concat_half(rem_Hi, rem_Wi)}};
  task->data.cfg.subtile = subtile;
}

inline void ne16_task_set_padding(ne16_task_t *task, const uint8_t top,
                                  const uint8_t bottom, const uint8_t left,
                                  const uint8_t right, const uint8_t value) {
  task->data.cfg.padding = ((top & 0xf) << 28) | ((right & 0xf) << 24) |
                           ((bottom & 0xf) << 20) | ((left & 0xf) << 16) |
                           (value & 0xff);
}

inline void ne16_task_set_mask_filter(ne16_task_t *task, const uint8_t top,
                                      const uint8_t right, const uint8_t bottom,
                                      const uint8_t left) {
  task->data.cfg.filter_mask = ((top & 0xff) << 24) | ((right & 0xff) << 16) |
                               ((bottom & 0xff) << 8) | ((left & 0xff) << 0);
}

void ne16_task_set_dims(ne16_task_t *task, const uint32_t w_in,
                        const uint32_t k_in, const uint32_t w_in_stride,
                        const uint32_t k_in_stride, const uint32_t h_out,
                        const uint32_t w_out, const uint32_t k_out,
                        const uint32_t w_out_stride, const uint32_t k_out_stride,
                        const uint8_t padding_top, const uint8_t padding_bottom,
                        const uint8_t padding_right,
                        const uint8_t padding_left) {
  ne16_task_set_strides(task, k_in, w_in_stride, k_in_stride, w_out_stride,
                        k_out_stride);
  ne16_task_set_counters(task, k_in, h_out, w_out, k_out, padding_bottom,
                         padding_right);
  ne16_task_set_padding(task, padding_top, padding_bottom, padding_left,
                        padding_right, 0);
}

void ne16_task_set_dims_stride2x2(
    ne16_task_t *task, const uint32_t h_in, const uint32_t w_in,
    const uint32_t k_in, const uint32_t w_in_stride, const uint32_t k_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t w_out_stride, const uint32_t k_out_stride,
    const uint8_t h_ker, const uint8_t w_ker, const uint8_t padding_top,
    const uint8_t padding_bottom, const uint8_t padding_right,
    const uint8_t padding_left) {
  const uint8_t stride = 2;

  ne16_task_set_strides(task, k_in, w_in_stride, k_in_stride, w_out_stride,
                        k_out_stride);
  ne16_task_set_counters(task, k_in, h_out > 1 ? 3 : 1, w_out > 1 ? 3 : 1,
                         k_out, h_in + padding_top >= 5 ? 0 : padding_bottom, 0);

  const uint8_t padding_bottom_new =
      (h_in + padding_top - h_ker) % stride == 0 ? 0 : padding_bottom;
  const uint8_t padding_right_new =
      (w_in + padding_left - w_ker) % stride == 0 ? 0 : padding_right;

  ne16_task_set_padding(task, padding_top, padding_bottom_new, padding_left,
                        padding_right_new, 0);
}
