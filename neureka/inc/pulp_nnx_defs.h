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

#ifndef __NEUREKA_DEFS_H__
#define __NEUREKA_DEFS_H__

/* ARHITECTURE */

#define NEUREKA_FILTER_SIZE (6)
#define NEUREKA_FILTER_BUFFER_SIZE (8)
#define NEUREKA_INPUT_CHANNEL_THROUGHPUT (32)
#define NEUREKA_INPUT_CHANNEL_THROUGHPUT_3x3 (28)
#define NEUREKA_OUTPUT_CHANNEL_THROUGHPUT (32)
#define NEUREKA_CONTEXT_SIZE (2)
#define NEUREKA_WEIGHT_BANDWIDTH (256)

#define NEUREKA_WEIGHT_D0_STRIDE_MODE8 (NEUREKA_INPUT_CHANNEL_THROUGHPUT / 8)
#define NEUREKA_WEIGHT_D0_STRIDE_MODE8_3x3 (NEUREKA_WEIGHT_BANDWIDTH / 8)
#define NEUREKA_WEIGHT_D0_STRIDE_MODE16 (NEUREKA_INPUT_CHANNEL_THROUGHPUT / 16)

/* REGISTER MAP */

#define NEUREKA_EVT0 12
#define NEUREKA_EVT1 13
#define NEUREKA_BASE_ADDR 0x00201000
#define WEIGHT_MEM_BASE 0x10400000
#define SRAM_OFFSET 0x00400000
#define MRAM_OFFSET 0x00000000

// Cluster
#define CLUSTER_CTRL_BASE_ADDR 0x00200000
#define CLUSTER_CTRL_HWPE_OFFS 0x18
#define CLUSTER_CTRL_HWPE_CG_EN_MASK 0x800

/* REGISTER OFFSETS */

// commands
#define NEUREKA_TRIGGER 0x00
#define NEUREKA_ACQUIRE 0x04
#define NEUREKA_FINISHED 0x08
#define NEUREKA_STATUS 0x0C
#define NEUREKA_RUNNING_JOB 0x10
#define NEUREKA_SOFT_CLEAR 0x14
#define NEUREKA_SWSYNC 0x18
#define NEUREKA_URISCY_IMEM 0x1C

// job configuration
#define NEUREKA_REGISTER_OFFSET 0x20

#define NEUREKA_REG_WEIGHTS_PTR 0x00
#define NEUREKA_REG_INFEAT_PTR 0x04
#define NEUREKA_REG_OUTFEAT_PTR 0x08
#define NEUREKA_REG_SCALE_PTR 0x0C
#define NEUREKA_REG_SCALE_SHIFT_PTR 0x10
#define NEUREKA_REG_SCALE_BIAS_PTR 0x14
#define NEUREKA_REG_INFEAT_D0_STRIDE 0x18
#define NEUREKA_REG_INFEAT_D1_STRIDE 0x1C
#define NEUREKA_REG_INFEAT_D2_STRIDE 0x20
#define NEUREKA_REG_OUTFEAT_D0_STRIDE 0x24
#define NEUREKA_REG_OUTFEAT_D1_STRIDE 0x28
#define NEUREKA_REG_OUTFEAT_D2_STRIDE 0x2C
#define NEUREKA_REG_WEIGHTS_D0_STRIDE 0x30
#define NEUREKA_REG_WEIGHTS_D1_STRIDE 0x34
#define NEUREKA_REG_WEIGHTS_D2_STRIDE 0x38
#define NEUREKA_REG_SUBTILE_REMAINDER_0 0x3C
#define NEUREKA_REG_SUBTILE_REMAINDER_1 0x40
#define NEUREKA_REG_SUBTILE_REMAINDER_2 0x44
#define NEUREKA_REG_SUBTILE_NUMBER_0 0x48
#define NEUREKA_REG_SUBTILE_NUMBER_1 0x4C
#define NEUREKA_REG_PADDING 0x50
#define NEUREKA_REG_WEIGHT_OFFSET_FACTOR 0x54
#define NEUREKA_REG_FILTER_MASKING 0x58
#define NEUREKA_REG_CONF0 0x5C

// Simulation only
#define NEUREKA_REG_GVSOC_TRACE 0x60

/*  SHIFT  */

#define NEUREKA_SHIFT_FLAG_NORM_BIAS (25)
#define NEUREKA_SHIFT_FLAG_NORM_SHIFT (24)
#define NEUREKA_SHIFT_QUANT_SHIFT (16)
#define NEUREKA_SHIFT_ROUNDING (11)

/*  CONF0 FLAGS */

#define NEUREKA_FLAG_NORM_BIAS (1 << 25)
#define NEUREKA_FLAG_NORM_SHIFT (1 << 24)
#define NEUREKA_FLAG_QUANT_FUNCTION_IDENTITY (1 << 23)
#define NEUREKA_FLAG_QUANT_FUNCTION_RELU (0 << 23)
#define NEUREKA_QUANT_MODE_8BIT (0 << 21)
#define NEUREKA_QUANT_MODE_16BIT (1 << 21)
#define NEUREKA_QUANT_MODE_32BIT (2 << 21)
// conf0[20:16] - quantization shift amount
#define NEUREKA_FLAG_WEIGHT_OFFSET_SYMMETRIC (0 << 15)
#define NEUREKA_FLAG_WEIGHT_OFFSET_LAYER_WISE (1 << 15)
#define NEUREKA_FLAG_STREAMIN (1 << 14)
#define NEUREKA_NORM_MODE_8BIT (0 << 12)
#define NEUREKA_NORM_MODE_16BIT (1 << 12)
#define NEUREKA_NORM_MODE_32BIT (2 << 12)
#define NEUREKA_FLAG_ROUND (1 << 11)
#define NEUREKA_FLAG_ACTIVATION_PREFETCH (1 << 10)
#define NEUREKA_FLAG_USE_WMEM (1 << 9)
#define NEUREKA_FLAG_USE_TCDM (0 << 9)
#define NEUREKA_FLAG_STRIDED_MODE (1 << 8)
#define NEUREKA_FLAG_LINEAR_MODE (1 << 7)
#define NEUREKA_FLAG_MODE_3x3 (0 << 5)
#define NEUREKA_FLAG_MODE_3x3_DW (1 << 5)
#define NEUREKA_FLAG_MODE_1x1 (2 << 5)
#define NEUREKA_FLAG_NORM_QUANT (1 << 4)
#define NEUREKA_FLAG_MODE_BASIC (0 << 3)
#define NEUREKA_FLAG_MODE16 (1 << 3)

/* Masks */

#define NEUREKA_MASK_QUANT_FUNCTION (1 << 23)
#define NEUREKA_MASK_QUANT_MODE (3 << 21)

/* Miscellaneous */

// Padding
#define MAX_PAD (0xf)

// Normalization
#define NEUREKA_NORM_MAX_LEN (32)
#define NO_NORM(length)                                                        \
  {                                                                            \
    .scale = scale_identity, .bias = NEUREKA_NULL, .shift = NEUREKA_NULL,      \
    .length = length, .mode = normMode32Bit                                    \
  }

// Quantization
#define NO_QUANT                                                               \
  {                                                                            \
    .shift_amount = 0, .mode = quantMode32Bit,                                 \
    .function = quantFunctionIdentity                                          \
  }

// GVSOC trace levels
#define NEUREKA_TRACE_LEVEL_JOB_START_END 0
#define NEUREKA_TRACE_LEVEL_CONFIG 1
#define NEUREKA_TRACE_LEVEL_ACTIV_INOUT 2
#define NEUREKA_TRACE_LEVEL_ALL 3

// null
#define NEUREKA_NULL ((void *)0)
#define NEUREKA_STATUS_FULL (0x101)

#endif // __NEUREKA_DEFS_H__
