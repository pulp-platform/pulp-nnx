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

#include "pulp_nnx_hal.h"
#include "pmsis.h"

static int qw, weight_d0_stride, outbytes;

// TODO For all the following functions we use __builtin_pulp_OffsetedWrite and
// __builtin_pulp_OffsetedRead instead of classic load/store because otherwise
// the compiler is not able to correctly factorize the NEUREKA base in case
// several accesses are done, ending up with twice more code

// __builtin_pulp_OffsetedX not defined - needs further investigation... (too
// old PULP toolchain? used v1.0.16) It is used inside PULP-SDK...

int nnx_empty() { return !NEUREKA_READ(NEUREKA_STATUS); }

int nnx_full() { return NEUREKA_READ(NEUREKA_STATUS) == NEUREKA_STATUS_FULL; }

int nnx_job_id() { return NEUREKA_READ(NEUREKA_RUNNING_JOB); }

void nnx_soft_clear() {
  NEUREKA_WRITE(NEUREKA_SOFT_CLEAR, 0);
  for (volatile int i = 0; i < 10; i++)
    ;
}

int nnx_acquire() {
  int job_id = -1;
  NEUREKA_BARRIER_ACQUIRE(job_id);
  return job_id;
}

void nnx_offload(nnx_task_t *task) {
  int *task_data = (int *)task;
  for (int i = 0; i < sizeof(nnx_task_t) / 4; ++i) {
    NEUREKA_WRITE_IO_REG(i * 4, task_data[i]);
  }
}

void nnx_offload_ptr(nnx_task_t *task) {
  int *task_data = (int *)task;
  for (int i = 0; i < 6; ++i) {
    NEUREKA_WRITE_IO_REG(i * 4, task_data[i]);
  }
}

void nnx_run_async() { NEUREKA_WRITE(NEUREKA_TRIGGER, 0); }

void nnx_run_blocking() {
  nnx_run_async();
  nnx_wait_empty();
}

void nnx_commit() {
  NEUREKA_WRITE(NEUREKA_TRIGGER, 1); // commit, no trigger
}

void nnx_busywait() { NEUREKA_BUSYWAIT(); }

void nnx_wait_empty() {
  while (!nnx_empty())
    NEUREKA_BARRIER_NOSTATUS();
}

void nnx_wait_not_full() {
  while (nnx_full())
    NEUREKA_BARRIER_NOSTATUS();
}

void nnx_wait_on_id(const int id) {
  while (nnx_job_id() <= id) {
    eu_evt_maskWaitAndClr(1 << NEUREKA_EVT0);
  };
}

void nnx_task_init(nnx_task_t *task) { memset(task, 0, sizeof(nnx_task_t)); }

int nnx_pad_input(nnx_cfg_t *cfg, const uint32_t top, const uint32_t right,
                  const uint32_t bottom, const uint32_t left,
                  const uint16_t value) {
  uint32_t padding = 0;
  uint32_t flags = 0;

  if (top > MAX_PAD || right > MAX_PAD || bottom > MAX_PAD || left > MAX_PAD) {
    return 1;
  }

  cfg->padding =
      (top << 28) + (right << 24) + (bottom << 20) + (left << 16) + value;

  return 0;
}

int nnx_norm_quant(nnx_cfg_t *cfg, const nnx_norm_t norm,
                   const nnx_quant_t quant) {
  if (quant.shift_amount > 31) {
    printf("ERROR! quant.shift_amount > 31\n");
    return 1;
  }

  if (quant.mode == quantMode16Bit) {
    printf("ERROR! quant.mode == quantMode16Bit\n");
    return 1;
  }

  BIT_SET(cfg->conf0, NEUREKA_FLAG_NORM_QUANT | quant.function | quant.mode |
                          (quant.shift_amount << 16) |
                          quant.flag_rounding << NEUREKA_SHIFT_ROUNDING |
                          norm.mode |
                          norm.flag_bias << NEUREKA_SHIFT_FLAG_NORM_BIAS |
                          norm.flag_shift << NEUREKA_SHIFT_FLAG_NORM_SHIFT);

  return 0;
}

void nnx_mask_filter(nnx_cfg_t *cfg, const uint8_t top, const uint8_t right,
                     const uint8_t bottom, const uint8_t left) {
  cfg->filter_mask = ((uint32_t)top << 24) | ((uint32_t)right << 16) |
                     ((uint32_t)bottom << 8) | ((uint32_t)left << 0);
}

nnx_error_code nnx_conv_1x1_update_dims(nnx_cfg_t *cfg, const int h_out,
                                        const int w_out, const int k_out,
                                        const int k_in) {

  const int num_Ko = divnceil(k_out, NEUREKA_OUTPUT_CHANNEL_THROUGHPUT);
  const int num_Ki = divnceil(k_in, NEUREKA_INPUT_CHANNEL_THROUGHPUT);
  const int num_Ho = divnceil(h_out, NEUREKA_FILTER_SIZE);
  const int num_Wo = divnceil(w_out, NEUREKA_FILTER_SIZE);

  const int rem_Ko = remainder(k_out, NEUREKA_OUTPUT_CHANNEL_THROUGHPUT);
  const int rem_Ki = remainder(k_in, NEUREKA_INPUT_CHANNEL_THROUGHPUT);
  const int rem_Ho = remainder(h_out, NEUREKA_FILTER_SIZE);
  const int rem_Wo = remainder(w_out, NEUREKA_FILTER_SIZE);
  const int rem_Hi = rem_Ho;
  const int rem_Wi = rem_Wo;

  const nnx_subtile_t subtile = {
      .number = {.KoKi = concat_half(num_Ko, num_Ki),
                 .HoWo = concat_half(num_Ho, num_Wo)},
      .remainder = {.KoKi = concat_half(rem_Ko, rem_Ki),
                    .HoWo = concat_half(rem_Ho, rem_Wo),
                    .HiWi = concat_half(rem_Hi, rem_Wi)}};
  cfg->subtile = subtile;

  // Strides
  const nnx_stride_t input_stride = {
      .d0 = k_in,
      .d1 = k_in * w_out,
      .d2 = k_in * 3 * 3 // copying arpan
  };
  cfg->input_stride = input_stride;

  const nnx_stride_t output_stride = {
      .d0 = 32, .d1 = k_out * outbytes, .d2 = k_out * outbytes * w_out};
  cfg->output_stride = output_stride;

  const nnx_stride_t weights_stride = {
      .d0 = weight_d0_stride * qw,
      .d1 = weight_d0_stride * qw * num_Ki,
      .d2 = 0 // Unused
  };
  cfg->weights_stride = weights_stride;

  return 0;
}

nnx_error_code nnx_conv_1x1(nnx_cfg_t *cfg, const nnx_weights_t weights,
                            const nnx_feature_t input,
                            const nnx_feature_t output) {
  if (weights.bitwidth < 2 || weights.bitwidth > 8) {
    return weightBitwidthOutOfBounds;
  }

  if (weights.offset_mode != weightOffsetModeLayerWise) {
    // Currently only layer-wise mode is used.
    return unsupportedWeightOffsetMode;
  }

  if ((input.bitwidth != featureBitwidth8Bit &&
       input.bitwidth != featureBitwidth16Bit) ||
      (output.bitwidth != featureBitwidth8Bit &&
       output.bitwidth != featureBitwidth32Bit)) {
    return unsupportedFeatureBitwidth;
  }

  if (input.height != output.height || input.width != output.width ||
      input.depth != weights.depth || output.depth != weights.n_weights) {
    return dimensionMismatch;
  }

  const int mode16 =
      input.bitwidth == 16 ? NEUREKA_FLAG_MODE16 : NEUREKA_FLAG_MODE_BASIC;

  BIT_SET(cfg->conf0, weights.offset_mode | NEUREKA_FLAG_MODE_1x1 | mode16 |
                          (weights.bitwidth - 1));

  // Global static variables needed by update_dims
  outbytes = output.bitwidth / 8;
  weight_d0_stride =
      mode16 ? NEUREKA_WEIGHT_D0_STRIDE_MODE16 : NEUREKA_WEIGHT_D0_STRIDE_MODE8;
  qw = weights.bitwidth;

  nnx_conv_1x1_update_dims(cfg, output.height, output.width, output.depth,
                           input.depth);

  // cfg->weight_offset_factor = SMALLEST_SIGNED(weights.bitwidth);
  cfg->weight_offset_factor = weights.offset_factor;

  return 0;
}

nnx_error_code nnx_conv_3x3_update_dims(nnx_cfg_t *cfg, const int h_out,
                                        const int w_out, const int k_out,
                                        const int k_in) {

  const int num_Ko = divnceil(k_out, NEUREKA_OUTPUT_CHANNEL_THROUGHPUT);
  const int num_Ki = divnceil(k_in, NEUREKA_INPUT_CHANNEL_THROUGHPUT_3x3);
  const int num_Ho = divnceil(h_out, NEUREKA_FILTER_SIZE);
  const int num_Wo = divnceil(w_out, NEUREKA_FILTER_SIZE);

  const int rem_Ko = remainder(k_out, NEUREKA_OUTPUT_CHANNEL_THROUGHPUT);
  const int rem_Ki = remainder(k_in, NEUREKA_INPUT_CHANNEL_THROUGHPUT_3x3);
  const int rem_Ho = remainder(h_out, NEUREKA_FILTER_SIZE);
  const int rem_Wo = remainder(w_out, NEUREKA_FILTER_SIZE);
  const int rem_Hi = rem_Ho + 2;
  const int rem_Wi = rem_Wo + 2;

  const nnx_subtile_t subtile = {
      .number = {.KoKi = concat_half(num_Ko, num_Ki),
                 .HoWo = concat_half(num_Ho, num_Wo)},
      .remainder = {.KoKi = concat_half(rem_Ko, rem_Ki),
                    .HoWo = concat_half(rem_Ho, rem_Wo),
                    .HiWi = concat_half(rem_Hi, rem_Wi)}};
  cfg->subtile = subtile;

  // Strides
  const nnx_stride_t input_stride = {.d0 = k_in,
                                     .d1 = k_in * (w_out + 2),
                                     .d2 = k_in * NEUREKA_FILTER_BUFFER_SIZE *
                                           NEUREKA_FILTER_BUFFER_SIZE};
  cfg->input_stride = input_stride;

  const nnx_stride_t output_stride = {
      .d0 = 32, .d1 = k_out * outbytes, .d2 = k_out * outbytes * w_out};
  cfg->output_stride = output_stride;

  const nnx_stride_t weights_stride = {
      .d0 = NEUREKA_WEIGHT_D0_STRIDE_MODE8_3x3,
      .d1 = NEUREKA_WEIGHT_D0_STRIDE_MODE8_3x3 * qw * num_Ki,
      .d2 = 0 // Unused
  };
  cfg->weights_stride = weights_stride;

  return 0;
}

nnx_error_code nnx_conv_3x3(nnx_cfg_t *cfg, const nnx_weights_t weights,
                            const nnx_feature_t input,
                            const nnx_feature_t output) {
  if (weights.bitwidth < 2 || weights.bitwidth > 8) {
    return weightBitwidthOutOfBounds;
  }

  if (weights.offset_mode != weightOffsetModeLayerWise) {
    // Currently only layer-wise mode is used.
    return unsupportedWeightOffsetMode;
  }

  if ((input.bitwidth != featureBitwidth8Bit &&
       input.bitwidth != featureBitwidth16Bit) ||
      (output.bitwidth != featureBitwidth8Bit &&
       output.bitwidth != featureBitwidth32Bit)) {
    return unsupportedFeatureBitwidth;
  }

  if (input.height - 2 != output.height || input.width - 2 != output.width ||
      input.depth != weights.depth || output.depth != weights.n_weights) {
    return dimensionMismatch;
  }

  const int mode16 =
      input.bitwidth == 16 ? NEUREKA_FLAG_MODE16 : NEUREKA_FLAG_MODE_BASIC;

  BIT_SET(cfg->conf0, weights.offset_mode | NEUREKA_FLAG_MODE_3x3 | mode16 |
                          (weights.bitwidth - 1));

  // Global static variables needed by update_dims
  outbytes = output.bitwidth / 8;
  weight_d0_stride =
      mode16 ? NEUREKA_WEIGHT_D0_STRIDE_MODE16 : NEUREKA_WEIGHT_D0_STRIDE_MODE8;
  qw = weights.bitwidth;

  nnx_conv_3x3_update_dims(cfg, output.height, output.width, output.depth,
                           input.depth);

  // cfg->weight_offset_factor = SMALLEST_SIGNED(weights.bitwidth);
  cfg->weight_offset_factor = weights.offset_factor;

  return 0;
}

nnx_error_code nnx_conv_3x3_dw_update_dims(nnx_cfg_t *cfg, const int h_out,
                                           const int w_out, const int k_out,
                                           const int k_in) {

  const int num_Ko = divnceil(k_out, NEUREKA_INPUT_CHANNEL_THROUGHPUT_3x3);
  const int num_Ki = num_Ko;
  const int num_Ho = divnceil(h_out, NEUREKA_FILTER_SIZE);
  const int num_Wo = divnceil(w_out, NEUREKA_FILTER_SIZE);

  const int rem_Ko = remainder(k_out, NEUREKA_INPUT_CHANNEL_THROUGHPUT_3x3);
  const int rem_Ki = rem_Ko;
  const int rem_Ho = remainder(h_out, NEUREKA_FILTER_SIZE);
  const int rem_Wo = remainder(w_out, NEUREKA_FILTER_SIZE);
  const int rem_Hi = rem_Ho + 2;
  const int rem_Wi = rem_Wo + 2;

  const nnx_subtile_t subtile = {
      .number = {.KoKi = concat_half(num_Ko, num_Ki),
                 .HoWo = concat_half(num_Ho, num_Wo)},
      .remainder = {.KoKi = concat_half(rem_Ko, rem_Ki),
                    .HoWo = concat_half(rem_Ho, rem_Wo),
                    .HiWi = concat_half(rem_Hi, rem_Wi)}};
  cfg->subtile = subtile;

  // Strides
  const nnx_stride_t input_stride = {
      .d0 = k_out,
      .d1 = k_out * (w_out + 2),
      .d2 = 0 // Unused
  };
  cfg->input_stride = input_stride;

  const nnx_stride_t output_stride = {
      .d0 = 32, .d1 = k_out * outbytes, .d2 = k_out * outbytes * w_out};
  cfg->output_stride = output_stride;

  const nnx_stride_t weights_stride = {
      .d0 = NEUREKA_FILTER_SIZE * NEUREKA_FILTER_SIZE * weight_d0_stride,
      .d1 = 0,
      .d2 = 0 // Unused
  };
  cfg->weights_stride = weights_stride;

  return 0;
}

nnx_error_code nnx_conv_3x3_dw(nnx_cfg_t *cfg, const nnx_weights_t weights,
                               const nnx_feature_t input,
                               const nnx_feature_t output) {
  if (weights.bitwidth < 2 || weights.bitwidth > 8) {
    return weightBitwidthOutOfBounds;
  }

  if (weights.offset_mode != weightOffsetModeLayerWise) {
    // Currently only layer-wise mode is used.
    return unsupportedWeightOffsetMode;
  }

  if ((input.bitwidth != featureBitwidth8Bit &&
       input.bitwidth != featureBitwidth16Bit) ||
      (output.bitwidth != featureBitwidth8Bit &&
       output.bitwidth != featureBitwidth32Bit)) {
    return unsupportedFeatureBitwidth;
  }

  if (input.height - 2 != output.height || input.width - 2 != output.width ||
      input.depth != output.depth) {
    return dimensionMismatch;
  }

  const int mode16 =
      input.bitwidth == 16 ? NEUREKA_FLAG_MODE16 : NEUREKA_FLAG_MODE_BASIC;

  BIT_SET(cfg->conf0, weights.offset_mode | NEUREKA_FLAG_MODE_3x3_DW | mode16 |
                          (weights.bitwidth - 1));

  // Global static variables needed by update_dims
  outbytes = output.bitwidth / 8;
  weight_d0_stride =
      mode16 ? NEUREKA_WEIGHT_D0_STRIDE_MODE16 : NEUREKA_WEIGHT_D0_STRIDE_MODE8;
  qw = weights.bitwidth;

  nnx_conv_3x3_dw_update_dims(cfg, output.height, output.width, output.depth,
                              input.depth);

  // cfg->weight_offset_factor = SMALLEST_SIGNED(weights.bitwidth);
  cfg->weight_offset_factor = weights.offset_factor;

  return 0;
}
