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

#ifndef __PULP_NNX_H__
#define __PULP_NNX_H__

#include <stdint.h>

typedef struct nnx_task_t nnx_task_t;
typedef struct nnx_norm_t nnx_norm_t;
typedef struct nnx_quant_t nnx_quant_t;
typedef enum nnx_weight_offset_mode_e nnx_weight_offset_mode_e;

void nnx_init(uint32_t max_stall);
void nnx_term();
int nnx_dispatch_check();
void nnx_dispatch_check_blocking();
void nnx_dispatch_task(nnx_task_t *task);
int nnx_resolve_check(nnx_task_t *task);
void nnx_resolve_check_blocking(nnx_task_t *task);

void nnx_task_init(nnx_task_t *task, const uint8_t kernel_shape,
                   const uint8_t depthwise, const uint8_t input_bits,
                   const uint8_t output_bits, const uint8_t weights_bits,
                   nnx_weight_offset_mode_e weights_offset_mode,
                   const uint32_t weights_offset_factor, nnx_quant_t quant,
                   nnx_norm_t norm, const uint8_t stride);
uint32_t nnx_pad_ptr(uint32_t ptr, const uint32_t width, const uint32_t channel,
                     const uint8_t bits, const uint8_t padding_top,
                     const uint8_t padding_left);
void nnx_task_set_ptrs(nnx_task_t *task, uint32_t input_ptr, uint32_t w_in,
                       uint32_t k_in, uint8_t bits_in, uint8_t padding_top,
                       uint8_t padding_left, uint32_t output_ptr,
                       uint32_t weights_ptr, uint32_t scale_ptr,
                       uint32_t shift_ptr, uint32_t bias_ptr);
void nnx_task_set_dims(nnx_task_t *task, const uint32_t w_in,
                       const uint32_t k_in, const uint32_t w_in_stride,
                       const uint32_t k_in_stride, const uint32_t h_out,
                       const uint32_t w_out, const uint32_t k_out,
                       const uint32_t w_out_stride, const uint32_t k_out_stride,
                       const uint8_t padding_top, const uint8_t padding_bottom,
                       const uint8_t padding_right, const uint8_t padding_left);
void nnx_task_set_dims_stride2x2(
    nnx_task_t *task, const uint32_t h_in, const uint32_t w_in,
    const uint32_t k_in, const uint32_t w_in_stride, const uint32_t k_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t w_out_stride, const uint32_t k_out_stride,
    const uint8_t h_ker, const uint8_t w_ker, const uint8_t padding_top,
    const uint8_t padding_bottom, const uint8_t padding_right,
    const uint8_t padding_left);
void nnx_dispatch_task_stride2x2(
    nnx_task_t *task, const uint32_t w_in, const uint32_t k_in,
    const uint32_t w_in_stride, const uint32_t k_in_stride,
    const uint32_t h_out, const uint32_t w_out, const uint32_t k_out,
    const uint32_t w_out_stride, const uint32_t k_out_stride,
    const uint8_t h_ker, const uint8_t w_ker);

#endif // __PULP_NNX_H__
