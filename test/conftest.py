# Luka Macan <luka.macan@unibo.it>
#
# Copyright 2023 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Union

import pytest
import pydantic
from Ne16MemoryLayout import Ne16MemoryLayout
from Ne16TestConf import Ne16TestConf
from NeurekaMemoryLayout import NeurekaMemoryLayout
from NeurekaTestConf import NeurekaTestConf
from NnxTestClasses import NnxTest, NnxTestGenerator


_SUPPORTED_ACCELERATORS = ["ne16", "neureka"]


def pytest_addoption(parser):
    # Lowercase shortoptions are reserved for pytest
    parser.addoption(
        "-T",
        "--test-dir",
        action="append",
        type=str,
        dest="test_dirs",
        required=True,
        help="Path to the a test directory.",
    )
    parser.addoption(
        "-R",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively search for tests in given test directories.",
    )
    parser.addoption(
        "-A",
        "--accelerator",
        choices=_SUPPORTED_ACCELERATORS,
        default="ne16",
        help="Choose an accelerator to test. Default: ne16",
    )
    parser.addoption(
        "--regenerate",
        action="store_true",
        default=False,
        help="Save the generated test data to their respective folders.",
    )
    parser.addoption(
        "--timeout",
        type=int,
        default=120,
        help="Execution timeout in seconds. Default: 120s",
    )


def _find_test_dirs(path: Union[str, os.PathLike]):
    return [dirpath for dirpath, _, _ in os.walk(path) if NnxTest.is_test_dir(dirpath)]


def pytest_generate_tests(metafunc):
    test_dirs = metafunc.config.getoption("test_dirs")
    recursive = metafunc.config.getoption("recursive")
    regenerate = metafunc.config.getoption("regenerate")
    timeout = metafunc.config.getoption("timeout")
    nnxName = metafunc.config.getoption("accelerator")

    if nnxName == "ne16":
        nnxMemoryLayoutCls = Ne16MemoryLayout
        nnxTestConfCls = Ne16TestConf
    elif nnxName == "neureka":
        nnxMemoryLayoutCls = NeurekaMemoryLayout
        nnxTestConfCls = NeurekaTestConf
    else:
        assert (
            False
        ), f"Given accelerator {nnxName} not supported. Supported accelerators: {_SUPPORTED_ACCELERATORS}"

    if recursive:
        tests_dirs = test_dirs
        test_dirs = []
        for tests_dir in tests_dirs:
            test_dirs.extend(_find_test_dirs(tests_dir))

    # Load valid tests
    nnxTestAndNames = []
    for test_dir in test_dirs:
        try:
            test = NnxTest.load(nnxTestConfCls, test_dir)
            # (Re)generate data
            if not test.is_valid() or regenerate:
                test = NnxTestGenerator.from_conf(test.conf, nnxMemoryLayoutCls.ACCUMULATOR_TYPE)
                test.save_data(test_dir)
            nnxTestAndNames.append((test, test_dir))
        except pydantic.ValidationError as e:
            _ = e
            nnxTestAndNames.append(
                pytest.param(
                    (None, test_dir),
                    marks=pytest.mark.skipif(
                        True, reason=f"Invalid test {test_dir}: {e.errors}"
                    ),
                )
            )

    metafunc.parametrize("nnxTestAndName", nnxTestAndNames)
    metafunc.parametrize("timeout", [timeout])
    metafunc.parametrize("nnxName", [nnxName])
    metafunc.parametrize("nnxMemoryLayoutCls", [nnxMemoryLayoutCls])
