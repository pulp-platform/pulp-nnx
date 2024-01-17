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

#include "pulp_nnx_neureka.h"
#include "hwpe.h"
#include "neureka.h"
#include "pulp_nnx_util.h"
#include <pmsis.h>
#include <stdint.h>
#include <sys/types.h>

void neureka_nnx_init(neureka_dev_t *dev, neureka_siracusa_conf_t *conf) {
  neureka_siracusa_open(conf);
  hwpe_soft_clear(&dev->hwpe_dev);
}

void neureka_nnx_term(neureka_dev_t *dev) {
  hwpe_soft_clear(&dev->hwpe_dev);
  neureka_siracusa_close();
}

int neureka_nnx_dispatch_check(neureka_dev_t *dev) {
  return !neureka_task_queue_full(dev);
}

void neureka_nnx_dispatch_wait(neureka_dev_t *dev) {
  while (!neureka_nnx_dispatch_check(dev)) {
    neureka_siracusa_event_wait_and_clear();
  }
}

int neureka_nnx_dispatch(neureka_dev_t *dev, neureka_task_t *task) {
  if (hwpe_task_queue_acquire_task(&dev->hwpe_dev, &task->id)) {
    return 1;
  }
  hwpe_task_queue_write_task(&dev->hwpe_dev, (uint32_t *)&task->data,
                             (int)(sizeof(neureka_task_data_t) / 4));
  hwpe_task_queue_release_and_run(&dev->hwpe_dev);
  return 0;
}

int neureka_nnx_resolve_check(neureka_dev_t *dev, neureka_task_t *task) {
#if __PLATFORM__ == ARCHI_PLATFORM_GVSOC
  // GVSOC model has a broken running_id so resolve_check
  // conservativly looks if the task queue is empty.
  return neureka_task_queue_empty(dev);
#else
  uint8_t prev_task_id = task->id - 1;
  return !(hwpe_last_task_id(&dev->hwpe_dev) == prev_task_id ||
           (hwpe_last_task_id(&dev->hwpe_dev) == task->id &&
            !neureka_task_queue_empty(dev)));
#endif
}

void neureka_nnx_resolve_wait(neureka_dev_t *dev, neureka_task_t *task) {
  while (!neureka_nnx_resolve_check(dev, task)) {
    neureka_siracusa_event_wait_and_clear();
  }
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

void neureka_nnx_dispatch_stride2x2(
    neureka_dev_t *dev, neureka_task_t *task, const uint32_t w_in,
    const uint32_t k_in, const uint32_t w_in_stride, const uint32_t k_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t w_out_stride, const uint32_t k_out_stride,
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
      task->data.infeat_ptr =
          _get_tile_ptr(input_base, i, j, 3 + h_ker - 1, 3 + w_ker - 1, k_in,
                        w_in_stride, k_in_stride, h_ker - stride,
                        w_ker - stride, i == 0 ? 0 : input_height_offset,
                        j == 0 ? 0 : input_width_offset, bits);
      task->data.outfeat_ptr =
          _get_tile_ptr(output_base, i, j, 2, 2, k_out, w_out_stride,
                        k_out_stride, 0, 0, i == 0 ? 0 : output_height_offset,
                        j == 0 ? 0 : output_width_offset, bits);

      task->data.cfg.padding =
          neureka_get_tile_padding(tile_padding, i, j, n_h, n_w);

      // Altered dispatch to wait if cannot acquire
      while (neureka_nnx_dispatch(dev, task)) {
        neureka_siracusa_event_wait_and_clear();
      }
    }
  }
}