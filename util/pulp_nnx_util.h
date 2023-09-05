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

#ifndef __NNX_UTIL_H__
#define __NNX_UTIL_H__

#include <stdint.h>

/**
 * divnceil
 *
 * Does integer division and ceiling of it.
 */
int divnceil(const int dividend, const int divisor);

/**
 * remainder
 *
 * Calculates the remainder but if the remainder should be 0,
 * returns divisor. Used for calculation of the last `remainding`
 * iteration of the tile.
 */
int remainder(const int dividend, const int divisor);

/**
 * concat_half
 *
 * Concatenate 2 16-bit numbers into a 32-bit number.
 */
uint32_t concat_half(const uint16_t high, const uint16_t low);

#endif // __NNX_UTIL_H__
