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
import subprocess
from typing import Union

import pydantic
import pytest

from NnxBuildFlow import CmakeBuildFlow, NnxBuildFlowName
from NnxMapping import NnxMapping, NnxName
from NnxTestClasses import NnxTest, NnxTestGenerator
from TestClasses import implies


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
        type=NnxName,
        choices=list(NnxName),
        default=NnxName.ne16,
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
    parser.addoption(
        "--build-flow",
        dest="buildFlowName",
        type=NnxBuildFlowName,
        choices=list(NnxBuildFlowName),
        default=NnxBuildFlowName.make,
        help="Choose the build flow. Default: make",
    )


@pytest.fixture
def timeout(request) -> int:
    return request.config.getoption("--timeout")


@pytest.fixture
def nnxName(request) -> NnxName:
    return request.config.getoption("--accelerator")


@pytest.fixture
def buildFlowName(request) -> NnxBuildFlowName:
    nnxName = request.config.getoption("--accelerator")
    buildFlowName = request.config.getoption("buildFlowName")

    assert implies(
        buildFlowName == NnxBuildFlowName.cmake, nnxName == NnxName.neureka_v2
    ), "The cmake build flow has been tested only with the neureka_v2 accelerator"

    if buildFlowName == NnxBuildFlowName.cmake:
        CmakeBuildFlow(nnxName).prepare()

    return buildFlowName


def _find_test_dirs(path: Union[str, os.PathLike]):
    return [dirpath for dirpath, _, _ in os.walk(path) if NnxTest.is_test_dir(dirpath)]


def pytest_generate_tests(metafunc):
    test_dirs = metafunc.config.getoption("test_dirs")
    recursive = metafunc.config.getoption("recursive")
    regenerate = metafunc.config.getoption("regenerate")
    nnxName = metafunc.config.getoption("accelerator")

    if recursive:
        tests_dirs = test_dirs
        test_dirs = []
        for tests_dir in tests_dirs:
            test_dirs.extend(_find_test_dirs(tests_dir))

    # Load valid tests
    nnxTestNames = []
    nnxTestConfCls = NnxMapping[nnxName].testConfCls
    for test_dir in test_dirs:
        try:
            test = NnxTest.load(nnxTestConfCls, test_dir)
            # (Re)generate data
            if not test.is_valid() or regenerate:
                test = NnxTestGenerator.from_conf(test.conf)
                test.save_data(test_dir)
            nnxTestNames.append(test_dir)
        except pydantic.ValidationError as e:
            for error in e.errors():
                if error["type"] == "missing":
                    raise e

            nnxTestNames.append(
                pytest.param(
                    test_dir,
                    marks=pytest.mark.skipif(
                        True, reason=f"Invalid test {test_dir}: {e.errors}"
                    ),
                )
            )

    metafunc.parametrize("nnxTestName", nnxTestNames)
