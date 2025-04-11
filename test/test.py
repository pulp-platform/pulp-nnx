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

import re

from NnxBuildFlow import AppName, NnxBuildFlowClsMapping, NnxBuildFlowName
from NnxMapping import NnxMapping, NnxName
from NnxTestClasses import NnxTest, NnxTestHeaderGenerator, NnxWmem

HORIZONTAL_LINE = "\n" + "-" * 100 + "\n"


def assert_message(msg: str, test_name: str, stdout: str):
    return (
        f"Test {test_name} failed: {msg}\n"
        + HORIZONTAL_LINE
        + f"\nCaptured stdout:\n{stdout}\n"
    )


def test(
    nnxName: NnxName,
    buildFlowName: NnxBuildFlowName,
    wmem: NnxWmem,
    nnxTestName: str,
    appName: AppName,
):
    testConfCls, weightCls = NnxMapping[nnxName]

    # conftest.py makes sure the test is valid and generated
    nnxTest = NnxTest.load(testConfCls, nnxTestName)

    NnxTestHeaderGenerator(weightCls(wmem), f"{appName.path()}/gen").generate(nnxTestName, nnxTest)

    buildFlow = NnxBuildFlowClsMapping[buildFlowName](nnxName, appName)
    buildFlow.build()
    stdout = buildFlow.run()

    match_success = re.search(r"> Success! No errors found.", stdout)
    match_fail = re.search(r"> Failure! Found (\d*)/(\d*) errors.", stdout)

    assert match_success or match_fail, assert_message(
        "No regexes matched.", nnxTestName, stdout
    )

    assert not match_fail, assert_message(
        f"Errors found: {match_fail.group(1)}/{match_fail.group(2)}",
        nnxTestName,
        stdout,
    )
