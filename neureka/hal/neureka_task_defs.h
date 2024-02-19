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

#ifndef __NEUREKA_DEFS_H__
#define __NEUREKA_DEFS_H__

/* ARCHITECTURE */

#define NNX_NEUREKA_PE_H (6)
#define NNX_NEUREKA_PE_W (6)

#define NEUREKA_SUBTILE_INPUT_HEIGHT_1x1 (NNX_NEUREKA_PE_H)
#define NEUREKA_SUBTILE_INPUT_WIDTH_1x1  (NNX_NEUREKA_PE_W)
#define NEUREKA_SUBTILE_INPUT_CHANNEL_1x1 (32)

#define NEUREKA_SUBTILE_INPUT_HEIGHT_3x3 (NNX_NEUREKA_PE_H+2)
#define NEUREKA_SUBTILE_INPUT_WIDTH_3x3  (NNX_NEUREKA_PE_W+2)
#define NEUREKA_SUBTILE_INPUT_CHANNEL_3x3 (32)

#define NEUREKA_SUBTILE_OUTPUT_HEIGHT (4)
#define NEUREKA_SUBTILE_OUTPUT_WIDTH (4)
#define NEUREKA_SUBTILE_OUTPUT_CHANNEL (32)

#define NEUREKA_OUTPUT_BANDWIDTH_BYTES (32)
#define NEUREKA_WEIGHT_BANDWIDTH_BYTES (32)

/* TASK REGISTERS */

// job configuration
#define NEUREKA_REG_WEIGHTS_PTR 0
#define NEUREKA_REG_INFEAT_PTR 1
#define NEUREKA_REG_OUTFEAT_PTR 2
#define NEUREKA_REG_SCALE_PTR 3
#define NEUREKA_REG_SCALE_SHIFT_PTR 4
#define NEUREKA_REG_SCALE_BIAS_PTR 5
#define NEUREKA_REG_INFEAT_D0_STRIDE 6
#define NEUREKA_REG_INFEAT_D1_STRIDE 7
#define NEUREKA_REG_INFEAT_D2_STRIDE 8
#define NEUREKA_REG_OUTFEAT_D0_STRIDE 9
#define NEUREKA_REG_OUTFEAT_D1_STRIDE 10
#define NEUREKA_REG_OUTFEAT_D2_STRIDE 11
#define NEUREKA_REG_WEIGHTS_D0_STRIDE 12
#define NEUREKA_REG_WEIGHTS_D1_STRIDE 13
#define NEUREKA_REG_WEIGHTS_D2_STRIDE 14
#define NEUREKA_REG_SUBTILE_REMAINDER_0 15
#define NEUREKA_REG_SUBTILE_REMAINDER_1 16
#define NEUREKA_REG_SUBTILE_REMAINDER_2 17
#define NEUREKA_REG_SUBTILE_NUMBER_0 18
#define NEUREKA_REG_SUBTILE_NUMBER_1 19
#define NEUREKA_REG_PADDING 20
#define NEUREKA_REG_WEIGHT_OFFSET_FACTOR 21
#define NEUREKA_REG_FILTER_MASKING 22
#define NEUREKA_REG_CONF0 23

/*  SHIFT  */

#define NEUREKA_SHIFT_FLAG_INPUT_SIGNED (26)
#define NEUREKA_SHIFT_FLAG_NORM_BIAS (25)
#define NEUREKA_SHIFT_FLAG_NORM_SHIFT (24)
#define NEUREKA_SHIFT_QUANT_SHIFT (16)

/*  CONF0 FLAGS */

#define NEUREKA_FLAG_INPUT_SIGNED (1 << 26)
#define NEUREKA_FLAG_NORM_BIAS (1 << 25)
#define NEUREKA_FLAG_NORM_SHIFT (1 << 24)
#define NEUREKA_FLAG_QUANT_FUNCTION_IDENTITY (1 << 23)
#define NEUREKA_FLAG_QUANT_FUNCTION_RELU (0 << 23)
#define NEUREKA_QUANT_MODE_8BIT (0 << 21)
#define NEUREKA_QUANT_MODE_32BIT (2 << 21)
// conf0[20:16] - quantization shift amount
#define NEUREKA_FLAG_WEIGHT_OFFSET_SYMMETRIC (0 << 15) // Unimplemented in gvsoc
#define NEUREKA_FLAG_WEIGHT_OFFSET_LAYER_WISE                                  \
  (1 << 15) // Unimplemented in gvsoc
#define NEUREKA_FLAG_STREAMIN (1 << 14)
#define NEUREKA_NORM_MODE_8BIT (0 << 12)
#define NEUREKA_NORM_MODE_32BIT (2 << 12)
#define NEUREKA_FLAG_ACTIVATION_PREFETCH (1 << 10)
#define NEUREKA_FLAG_WEIGHT_SOURCE_WMEM (1 << 9)
#define NEUREKA_FLAG_WEIGHT_SOURCE_TCDM (0 << 9)
#define NEUREKA_FLAG_LINEAR_MODE (1 << 7) // not tested
#define NEUREKA_FLAG_MODE_3x3 (0 << 5)
#define NEUREKA_FLAG_MODE_3x3_DW (1 << 5)
#define NEUREKA_FLAG_MODE_1x1 (2 << 5)
#define NEUREKA_FLAG_NORM_QUANT (1 << 4)

/* Masks */

#define NEUREKA_MASK_FLAG_INPUT_SIGNED (0x1 << 26)
#define NEUREKA_MASK_FLAG_NORM_BIAS (0x1 << 25)
#define NEUREKA_MASK_FLAG_NORM_SHIFT (0x1 << 24)
#define NEUREKA_MASK_QUANT_FUNCTION (0x1 << 23)
#define NEUREKA_MASK_QUANT_MODE (0x3 << 21)
#define NEUREKA_MASK_SHIFT_AMOUNT (0x1f << 16)
#define NEUREKA_MASK_WEIGHT_OFFSET_MODE (0x1 << 15)
#define NEUREKA_MASK_NORM_MODE (0x3 << 12)
#define NEUREKA_MASK_FLAG_ACTIVATION_PREFETCH (0x1 << 10)
#define NEUREKA_MASK_FLAG_WEIGHT_SOURCE (0x1 << 9)
#define NEUREKA_MASK_FLAG_MODE (0x3 << 5)
#define NEUREKA_MASK_FLAG_WEIGHT_BITS (0x7 << 0)

/* PADDING */

#define NEUREKA_DONT_PAD (0)
#define NEUREKA_MAX_PAD (2)

/* NORM */
#define NEUREKA_NORM_MAX_LEN (32)

#endif // __NEUREKA_DEFS_H__
