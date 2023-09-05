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
 */

#include "ne16_hal.h"
#include "pmsis.h"
#include "pulp_nnx.h"
#include "pulp_nnx_util.h"
#include <stdint.h>

inline void nnx_init(uint32_t max_stall) {
  ne16_cg_enable();
  ne16_setpriority_ne16();
  ne16_set_max_stall(max_stall);
  ne16_soft_clear();
}

inline void nnx_term() {
  ne16_soft_clear();
  ne16_setpriority_core();
  ne16_reset_max_stall();
  ne16_cg_disable();
}

/** nnx_dispatch_check
 *
 * Check whether you can dispatch to the accelerator.
 */
inline int nnx_dispatch_check() { return !ne16_full(); }

/** nnx_dispatch_check_blocking
 *
 * Block until you can dispatch to the accelerator.
 */
inline void nnx_dispatch_check_blocking() {
  while (!nnx_dispatch_check()) {
    ne16_event_wait();
  }
}

/** nnx_dispatch_task
 *
 * Dispatch a task to the accelerator, assuming it
 * was checked before.
 */
inline void nnx_dispatch_task(nnx_task_t *task) {
  task->id = ne16_acquire();
  ne16_task_offload(task);
  ne16_run_async();
}

/** nnx_resolve_check
 *
 * Check whether the task has been resolved.
 */
inline int nnx_resolve_check(nnx_task_t *task) {
  uint8_t prev_task_id = task->id - 1;
  return !(ne16_last_task_id() == prev_task_id ||
           (ne16_last_task_id() == task->id && !ne16_empty()));
}

/** nnx_resolve_check_blocking
 *
 * Block until you can resolve the task.
 */
inline void nnx_resolve_check_blocking(nnx_task_t *task) {
  while (!nnx_resolve_check(task)) {
    ne16_event_wait();
  }
}

inline void nnx_task_init(nnx_task_t *task, const uint8_t kernel_shape,
                          const uint8_t depthwise, const uint8_t input_bits,
                          const uint8_t output_bits, const uint8_t weights_bits,
                          nnx_weight_offset_mode_e weights_offset_mode,
                          const uint32_t weights_offset_factor,
                          nnx_quant_t quant, nnx_norm_t norm,
                          const uint8_t stride) {

  ne16_task_init(task, kernel_shape, depthwise, input_bits, output_bits,
                 weights_bits, weights_offset_mode, weights_offset_factor,
                 quant, norm, stride);
}

/** nnx_pad_ptr
 *
 * Calculate the pointer to the start of the ptr as if
 * it was the start to the padded data.
 * Necessary for input pointer when it's padded.
 */
inline uint32_t nnx_pad_ptr(uint32_t ptr, const uint32_t width,
                            const uint32_t channel, const uint8_t bits,
                            const uint8_t padding_top,
                            const uint8_t padding_left) {
  return ptr - (padding_top * width + padding_left) * channel * bits / 8;
}

inline void nnx_task_set_ptrs(nnx_task_t *task, uint32_t input_ptr,
                              uint32_t w_in, uint32_t k_in, uint8_t bits_in,
                              uint8_t padding_top, uint8_t padding_left,
                              uint32_t output_ptr, uint32_t weights_ptr,
                              uint32_t scale_ptr, uint32_t shift_ptr,
                              uint32_t bias_ptr) {
  task->data.infeat_ptr =
      nnx_pad_ptr(input_ptr, w_in, k_in, bits_in, padding_top, padding_left);
  task->data.outfeat_ptr = output_ptr;
  task->data.weights_ptr = weights_ptr;
  task->data.scale_ptr = scale_ptr;
  task->data.scale_shift_ptr = shift_ptr;
  task->data.scale_bias_ptr = bias_ptr;
}

void nnx_task_set_dims(nnx_task_t *task, const uint32_t w_in,
                       const uint32_t k_in, const uint32_t h_out,
                       const uint32_t w_out, const uint32_t k_out,
                       const uint8_t padding_top, const uint8_t padding_bottom,
                       const uint8_t padding_right,
                       const uint8_t padding_left) {
  ne16_task_set_strides(task, k_in, w_in, k_in, w_out, k_out);
  ne16_task_set_counters(task, k_in, h_out, w_out, k_out, padding_bottom,
                         padding_right);
  ne16_task_set_padding(task, padding_top, padding_bottom, padding_left,
                        padding_right, 0);
}

void nnx_task_set_dims_stride2x2(nnx_task_t *task, const uint32_t h_in,
                                 const uint32_t w_in, const uint32_t k_in,
                                 const uint32_t h_out, const uint32_t w_out,
                                 const uint32_t k_out, const uint8_t h_ker,
                                 const uint8_t w_ker, const uint8_t padding_top,
                                 const uint8_t padding_bottom,
                                 const uint8_t padding_right,
                                 const uint8_t padding_left) {
  const uint8_t stride = 2;

  ne16_task_set_strides(task, k_in, w_in, k_in, w_out, k_out);
  ne16_task_set_counters(task, k_in, h_out > 1 ? 3 : 1, w_out > 1 ? 3 : 1,
                         k_out, 0, 0);

  const uint8_t padding_bottom_new =
      (h_in + padding_top - h_ker) % stride == 0 ? 0 : padding_bottom;
  const uint8_t padding_right_new =
      (w_in + padding_left - w_ker) % stride == 0 ? 0 : padding_right;

  ne16_task_set_padding(task, padding_top, padding_bottom_new, padding_left,
                        padding_right_new, 0);
}

static inline uint32_t _get_tile_ptr(uint32_t ptr, int i, int j, int size_i,
                                     uint32_t size_j, uint32_t size_k,
                                     uint32_t stride_j, uint32_t stride_k,
                                     uint32_t overlap_i, uint32_t overlap_j,
                                     uint32_t offset_i, uint32_t offset_j,
                                     uint8_t data_size) {
  return ptr +
         (i * (size_i - overlap_i) - offset_i) * stride_j * stride_k *
             data_size / 8 +
         (j * (size_j - overlap_j) - offset_j) * stride_k * data_size / 8;
}

/** nnx_dispatch_task_stride2x2
 *
 * It uses NE16's 2x2 strided mode which reduces the number of writes NE16 does.
 * This mode doesn't stride the NE16's subtile input pointer, so we have to
 * tile the tile to the subtile's spatial dimensions (in this case 3x3 output).
 * Works only if the k_out is divisible by 2.
 */
void nnx_dispatch_task_stride2x2(nnx_task_t *task, const uint32_t w_in,
                                 const uint32_t k_in, const uint32_t h_out,
                                 const uint32_t w_out, const uint32_t k_out,
                                 const uint8_t h_ker, const uint8_t w_ker) {
  const uint8_t stride = 2;
  const uint8_t bits = 8;

  const uint32_t n_h = divnceil(h_out, stride);
  const uint32_t n_w = divnceil(w_out, stride);
  const uint32_t input_height_offset = h_out % stride == 1 ? stride : 0;
  const uint32_t input_width_offset = w_out % stride == 1 ? stride : 0;
  const uint32_t output_height_offset = h_out % stride == 1 ? 1 : 0;
  const uint32_t output_width_offset = w_out % stride == 1 ? 1 : 0;

  const uint32_t input_base = task->data.infeat_ptr;
  const uint32_t output_base = task->data.outfeat_ptr;
  const uint32_t tile_padding = task->data.cfg.padding;

  for (int i = 0; i < n_h; i++) {
    for (int j = 0; j < n_w; j++) {
      task->data.infeat_ptr = _get_tile_ptr(
          input_base, i, j, 3 + h_ker - 1, 3 + w_ker - 1, k_in, w_in, k_in,
          h_ker - stride, w_ker - stride, i == 0 ? 0 : input_height_offset,
          j == 0 ? 0 : input_width_offset, bits);
      task->data.outfeat_ptr =
          _get_tile_ptr(output_base, i, j, 2, 2, k_out, w_out, k_out, 0, 0,
                        i == 0 ? 0 : output_height_offset,
                        j == 0 ? 0 : output_width_offset, bits);

      task->data.cfg.padding =
          ne16_get_tile_padding(tile_padding, i, j, n_h, n_w);

      nnx_dispatch_check_blocking();
      nnx_dispatch_task(task);
    }
  }
}
