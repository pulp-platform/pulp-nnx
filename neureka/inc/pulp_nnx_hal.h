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

#ifndef __NEUREKA_H__
#define __NEUREKA_H__

#include <stdint.h>

#include "pulp_nnx_defs.h"
#include "pulp_nnx_error_codes.h"

#define NEUREKA_CG_ENABLE()                                                    \
  *(volatile int *)(CLUSTER_CTRL_BASE_ADDR + CLUSTER_CTRL_HWPE_OFFS) |=        \
      CLUSTER_CTRL_HWPE_CG_EN_MASK
#define NEUREKA_CG_DISABLE()                                                   \
  *(volatile int *)(CLUSTER_CTRL_BASE_ADDR + CLUSTER_CTRL_HWPE_OFFS) &=        \
      ~CLUSTER_CTRL_HWPE_CG_EN_MASK

#define NEUREKA_WRITE(offset, value)                                           \
  *(int volatile *)(NEUREKA_BASE_ADDR + (offset)) = (value)
#define NEUREKA_WRITE_BE(offset, value, be)                                    \
  *(char volatile *)(NEUREKA_BASE_ADDR + (offset) + (be)) = (value)
#define NEUREKA_READ(offset) *(int volatile *)(NEUREKA_BASE_ADDR + (offset))

#define NEUREKA_WRITE_IO_REG(offset, value)                                    \
  NEUREKA_WRITE(NEUREKA_REGISTER_OFFSET + (offset), (value))
#define NEUREKA_WRITE_IO_REG_BE(offset, value, be)                             \
  NEUREKA_WRITE_BE(NEUREKA_REGISTER_OFFSET + (offset), (value), (be))
#define NEUREKA_READ_IO_REG(offset)                                            \
  NEUREKA_READ(NEUREKA_REGISTER_OFFSET + (offset))

#define NEUREKA_BARRIER_NOSTATUS() eu_evt_maskWaitAndClr(1 << NEUREKA_EVT0)
#define NEUREKA_BARRIER()                                                      \
  do {                                                                         \
    eu_evt_maskWaitAndClr(1 << NEUREKA_EVT0);                                  \
  } while ((*(int volatile *)(NEUREKA_BASE_ADDR + NEUREKA_STATUS)) != 0)
#define NEUREKA_BUSYWAIT()                                                     \
  do {                                                                         \
  } while ((*(int volatile *)(NEUREKA_BASE_ADDR + NEUREKA_STATUS)) != 0)
#define NEUREKA_BARRIER_ACQUIRE(job_id)                                        \
  job_id = NEUREKA_READ(NEUREKA_ACQUIRE);                                      \
  while (job_id < 0) {                                                         \
    eu_evt_maskWaitAndClr(1 << NEUREKA_EVT0);                                  \
    job_id = NEUREKA_READ(NEUREKA_ACQUIRE);                                    \
  };
#define NEUREKA_NOBARRIER_ACQUIRE(job_id)                                      \
  job_id = NEUREKA_READ(NEUREKA_ACQUIRE);                                      \
  while (job_id < 0) {                                                         \
    job_id = NEUREKA_READ(NEUREKA_ACQUIRE);                                    \
  };

#define DIVNCEIL(A, B) (((A - 1) / B) + 1)
#define REMAINDER(A, B) (((A - 1) % B) + 1)
#define CONCAT_HALF(A, B) (((A & 0xffff) << 16) | (B & 0xffff))

#define NNX_CONTEXT_SIZE NEUREKA_CONTEXT_SIZE

#define FLAG_USED (1)
#define FLAG_UNUSED (0)

typedef enum {
  weightOffsetModeSymmetric = NEUREKA_FLAG_WEIGHT_OFFSET_SYMMETRIC,
  weightOffsetModeLayerWise = NEUREKA_FLAG_WEIGHT_OFFSET_LAYER_WISE
} nnx_weight_offset_mode_e;

typedef struct {
  void *data;
  uint16_t height;
  uint16_t width;
  uint16_t depth;
  uint16_t n_weights;
  uint32_t bitwidth;
  int32_t offset_factor;
  nnx_weight_offset_mode_e offset_mode;
} nnx_weights_t;

typedef enum {
  featureBitwidth8Bit = 8,
  featureBitwidth16Bit = 16,
  featureBitwidth32Bit = 32
} nnx_feature_bitwidth_e;

typedef struct {
  void *data;
  uint16_t height;
  uint16_t width;
  uint16_t depth;
  nnx_feature_bitwidth_e bitwidth;
} nnx_feature_t;

typedef enum {
  normMode8Bit = NEUREKA_NORM_MODE_8BIT,
  normMode16Bit = NEUREKA_NORM_MODE_16BIT,
  normMode32Bit = NEUREKA_NORM_MODE_32BIT
} nnx_norm_mode_e;

typedef struct {
  nnx_norm_mode_e mode;
  int flag_bias;
  int flag_shift;
} nnx_norm_t;

typedef enum {
  quantMode8Bit = NEUREKA_QUANT_MODE_8BIT,
  quantMode16Bit = NEUREKA_QUANT_MODE_16BIT,
  quantMode32Bit = NEUREKA_QUANT_MODE_32BIT
} nnx_quant_mode_e;

typedef enum {
  quantFunctionIdentity = NEUREKA_FLAG_QUANT_FUNCTION_IDENTITY,
  quantFunctionRelu = NEUREKA_FLAG_QUANT_FUNCTION_RELU
} nnx_quant_function_e;

// TODO: add rounding to quant. Should also be an enum? Best boolean...
typedef struct {
  // Shift amount must be in range 0x00-0x1F
  unsigned shift_amount;
  nnx_quant_mode_e mode;
  nnx_quant_function_e function;
  int flag_rounding;
} nnx_quant_t;

typedef struct {
  uint32_t d0;
  uint32_t d1;
  uint32_t d2;
} nnx_stride_t;

typedef struct {
  uint32_t KoKi;
  uint32_t HoWo;
  uint32_t HiWi;
} nnx_subtile_remainder_t;

typedef struct {
  uint32_t KoKi;
  uint32_t HoWo;
} nnx_subtile_number_t;

typedef struct {
  nnx_subtile_remainder_t remainder;
  nnx_subtile_number_t number;
} nnx_subtile_t;

typedef struct {
  nnx_stride_t input_stride;
  nnx_stride_t output_stride;
  nnx_stride_t weights_stride;
  nnx_subtile_t subtile;
  uint32_t padding;
  uint32_t weight_offset_factor;
  uint32_t filter_mask;
  uint32_t conf0;
} nnx_cfg_t;

typedef struct {
  uint32_t weights_ptr;
  uint32_t infeat_ptr;
  uint32_t outfeat_ptr;
  uint32_t scale_ptr;
  uint32_t scale_shift_ptr;
  uint32_t scale_bias_ptr;
  nnx_cfg_t cfg;
} nnx_task_t;

int nnx_job_id();
int nnx_empty();
int nnx_full();
void nnx_soft_clear();
int nnx_acquire();
void nnx_offload(nnx_task_t *task);
void nnx_offload_ptr(nnx_task_t *task);
void nnx_run_async();
void nnx_run_blocking();
void nnx_commit();
void nnx_wait_empty();
void nnx_wait_not_full();
void nnx_wait_on_id(int id);
void nnx_busywait();

void nnx_task_init(nnx_task_t *task);
int nnx_pad_input(nnx_cfg_t *cfg, uint32_t top, uint32_t right, uint32_t bottom,
                  uint32_t left, uint16_t value);
int nnx_norm_quant(nnx_cfg_t *cfg, nnx_norm_t norm, nnx_quant_t quant);
void nnx_mask_filter(nnx_cfg_t *cfg, uint8_t top, uint8_t right, uint8_t bottom,
                     uint8_t left);
nnx_error_code nnx_conv_1x1(nnx_cfg_t *cfg, nnx_weights_t weights,
                            nnx_feature_t input, nnx_feature_t output);
nnx_error_code nnx_conv_1x1_update_dims(nnx_cfg_t *cfg, int h_out, int w_out,
                                        int k_out, int k_in);
nnx_error_code nnx_conv_3x3(nnx_cfg_t *cfg, nnx_weights_t weights,
                            nnx_feature_t input, nnx_feature_t output);
nnx_error_code nnx_conv_3x3_update_dims(nnx_cfg_t *cfg, int h_out, int w_out,
                                        int k_out, int k_in);
nnx_error_code nnx_conv_3x3_dw(nnx_cfg_t *cfg, nnx_weights_t weights,
                               nnx_feature_t input, nnx_feature_t output);
nnx_error_code nnx_conv_3x3_dw_update_dims(nnx_cfg_t *cfg, int h_out, int w_out,
                                           int k_out, int k_in);

#endif /* __NEUREKA_H__ */
