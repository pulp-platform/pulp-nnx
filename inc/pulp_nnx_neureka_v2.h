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

#include "neureka_v2.h"
#include "neureka_v2_siracusa_bsp.h"
#include "neureka_v2_task.h"
#include <stdint.h>

/* PULP-NNX interface */

void neureka_v2_nnx_init(const neureka_v2_dev_t *dev,
                         neureka_v2_siracusa_conf_t *conf);
void neureka_v2_nnx_term(const neureka_v2_dev_t *dev);

/** neureka_v2_nnx_dispatch_check
 *
 * Check whether you can dispatch to the accelerator.
 */
int neureka_v2_nnx_dispatch_check(const neureka_v2_dev_t *dev);

/** neureka_v2_nnx_dispatch_wait
 *
 * Block until you can dispatch to the accelerator.
 */
void neureka_v2_nnx_dispatch_wait(const neureka_v2_dev_t *dev);

/** neureka_v2_nnx_dispatch
 *
 * Dispatch a task to the accelerator.
 * Fails with return code 1 if the task cannot be dispatched. Otherwise returns
 * 0.
 */
int neureka_v2_nnx_dispatch(const neureka_v2_dev_t *dev,
                            neureka_v2_task_t *task);

/** neureka_v2_nnx_resolve_check
 *
 * Check whether the task has been resolved.
 */
int neureka_v2_nnx_resolve_check(const neureka_v2_dev_t *dev,
                                 neureka_v2_task_t *task);

/** neureka_v2_nnx_resolve_wait
 *
 * Block until you can resolve the task.
 */
void neureka_v2_nnx_resolve_wait(const neureka_v2_dev_t *dev,
                                 neureka_v2_task_t *task);
