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

#include "pulp_nnx_util.h"

inline int divnceil(const int dividend, const int divisor) {
  return ((dividend - 1) / divisor) + 1;
}

inline int remainder(const int dividend, const int divisor) {
  return ((dividend - 1) % divisor) + 1;
}

inline uint32_t concat_half(const uint16_t high, const uint16_t low) {
  return ((uint32_t)high << 16) | low;
}
