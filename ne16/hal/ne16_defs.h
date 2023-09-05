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

#ifndef __NE16_DEFS_H__
#define __NE16_DEFS_H__

/* ARHITECTURE */

#define NE16_FILTER_SIZE (3)
#define NE16_FILTER_BUFFER_SIZE (5)
#define NE16_INPUT_CHANNEL_THROUGHPUT (16)
#define NE16_OUTPUT_CHANNEL_THROUGHPUT (32)
#define NE16_CONTEXT_SIZE (2)

#define NE16_WEIGHT_D0_STRIDE_MODE8 (2)
#define NE16_WEIGHT_D0_STRIDE_MODE16 (1)

/* REGISTER MAP */

#define NE16_EVT0 (1 << 12)
#define NE16_EVT1 (1 << 13)
#define NE16_BASE_ADDR (0x00201000)

/* CLUSTER */

#define CLUSTER_CTRL_ADDR_BASE (0x00200000)

/* CLUSTER_HWPE */

#define CLUSTER_CTRL_HWPE_OFFS 0x18

#define CLUSTER_CTRL_HWPE_ADDR (CLUSTER_CTRL_ADDR_BASE + CLUSTER_CTRL_HWPE_OFFS)

#define CLUSTER_CTRL_HWPE_MASK_CG_EN 0x800
#define CLUSTER_CTRL_HWPE_MASK_HCI_PRIO 0x100
#define CLUSTER_CTRL_HWPE_MASK_HCI_MAXSTALL 0xff

/* REGISTER OFFSETS */

// commands
#define NE16_TRIGGER 0x00
#define NE16_ACQUIRE 0x04
#define NE16_FINISHED 0x08
#define NE16_STATUS 0x0C
#define NE16_RUNNING_JOB 0x10
#define NE16_SOFT_CLEAR 0x14
#define NE16_SWSYNC 0x18
#define NE16_URISCY_IMEM 0x1C

// job configuration
#define NE16_REGISTER_OFFSET 0x20

#define NE16_REG_WEIGHTS_PTR 0x00
#define NE16_REG_INFEAT_PTR 0x04
#define NE16_REG_OUTFEAT_PTR 0x08
#define NE16_REG_SCALE_PTR 0x0C
#define NE16_REG_SCALE_SHIFT_PTR 0x10
#define NE16_REG_SCALE_BIAS_PTR 0x14
#define NE16_REG_INFEAT_D0_STRIDE 0x18
#define NE16_REG_INFEAT_D1_STRIDE 0x1C
#define NE16_REG_INFEAT_D2_STRIDE 0x20
#define NE16_REG_OUTFEAT_D0_STRIDE 0x24
#define NE16_REG_OUTFEAT_D1_STRIDE 0x28
#define NE16_REG_OUTFEAT_D2_STRIDE 0x2C
#define NE16_REG_WEIGHTS_D0_STRIDE 0x30
#define NE16_REG_WEIGHTS_D1_STRIDE 0x34
#define NE16_REG_WEIGHTS_D2_STRIDE 0x38
#define NE16_REG_SUBTILE_REMAINDER_0 0x3C
#define NE16_REG_SUBTILE_REMAINDER_1 0x40
#define NE16_REG_SUBTILE_REMAINDER_2 0x44
#define NE16_REG_SUBTILE_NUMBER_0 0x48
#define NE16_REG_SUBTILE_NUMBER_1 0x4C
#define NE16_REG_PADDING 0x50
#define NE16_REG_WEIGHT_OFFSET_FACTOR 0x54
#define NE16_REG_FILTER_MASKING 0x58
#define NE16_REG_CONF0 0x5C

/*  SHIFT  */

#define NE16_SHIFT_FLAG_NORM_BIAS (25)
#define NE16_SHIFT_FLAG_NORM_SHIFT (24)
#define NE16_SHIFT_ROUNDING (11)

/*  CONF0 FLAGS */

#define NE16_FLAG_NORM_BIAS (1 << 25)
#define NE16_FLAG_NORM_SHIFT (1 << 24)
#define NE16_FLAG_QUANT_FUNCTION_IDENTITY (1 << 23)
#define NE16_FLAG_QUANT_FUNCTION_RELU (0 << 23)
#define NE16_QUANT_MODE_8BIT (0 << 21)
#define NE16_QUANT_MODE_16BIT (1 << 21)
#define NE16_QUANT_MODE_32BIT (2 << 21)
// conf0[20:16] - quantization shift amount
#define NE16_FLAG_WEIGHT_OFFSET_SYMMETRIC (0 << 15)
#define NE16_FLAG_WEIGHT_OFFSET_LAYER_WISE (1 << 15)
#define NE16_FLAG_STREAMIN (1 << 14)
#define NE16_NORM_MODE_8BIT (0 << 12)
#define NE16_NORM_MODE_16BIT (1 << 12)
#define NE16_NORM_MODE_32BIT (2 << 12)
#define NE16_FLAG_ROUND (1 << 11)
#define NE16_FLAG_STRIDE_2x2 (1 << 8)
#define NE16_FLAG_LINEAR_MODE (1 << 7)
#define NE16_FLAG_MODE_3x3 (0 << 5)
#define NE16_FLAG_MODE_3x3_DW (1 << 5)
#define NE16_FLAG_MODE_1x1 (2 << 5)
#define NE16_FLAG_NORM_QUANT (1 << 4)
#define NE16_FLAG_MODE_BASIC (0 << 3)
#define NE16_FLAG_MODE16 (1 << 3)

/* Masks */

#define NE16_MASK_QUANT_FUNCTION (1 << 23)
#define NE16_MASK_QUANT_MODE (3 << 21)

/* PADDING */

#define NE16_DONT_PAD (0)
#define NE16_MAX_PAD (2)

/* NORM */
#define NE16_NORM_MAX_LEN (32)
#define NE16_NO_NORM(length)                                                   \
  {                                                                            \
    .scale = scale_identity, .bias = NE16_NULL, .shift = NE16_NULL,            \
    .length = length, .mode = normMode32Bit                                    \
  }

/* QUANT */
#define NE16_NO_QUANT                                                          \
  {                                                                            \
    .shift_amount = 0, .mode = quantMode32Bit,                                 \
    .function = quantFunctionIdentity                                          \
  }

/* NULL */
#define NE16_NULL ((void *)0)

#define NE16_STATUS_FULL (0x101)

#endif // __NE16_DEFS_H__
