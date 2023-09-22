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
from Ne16TestClasses import Ne16Test, Ne16TestGenerator


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
        "--regenerate",
        action="store_true",
        default=False,
        help="Save the generated test data to their respective folders.",
    )


def _find_test_dirs(path: Union[str, os.PathLike]):
    return [dirpath for dirpath, _, _ in os.walk(path) if Ne16Test.is_test_dir(dirpath)]


def pytest_generate_tests(metafunc):
    test_dirs = metafunc.config.getoption("test_dirs")
    recursive = metafunc.config.getoption("recursive")
    regenerate = metafunc.config.getoption("regenerate")

    if recursive:
        tests_dirs = test_dirs
        test_dirs = []
        for tests_dir in tests_dirs:
            test_dirs.extend(_find_test_dirs(tests_dir))

    # (Re)Generate test data
    for test_dir in test_dirs:
        test = Ne16Test.load(test_dir)
        if not test.is_valid() or regenerate:
            test = Ne16TestGenerator.from_conf(test.conf)
            test.save_data(test_dir)

    metafunc.parametrize("path", test_dirs)
