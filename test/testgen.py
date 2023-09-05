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

import os
import argparse
import json
import toml
from typing import Optional, Union, Set
from Ne16TestClasses import (
    Ne16TestConf,
    Ne16TestGenerator,
    Ne16Test,
    Ne16TestHeaderGenerator,
)


def headers_gen(args, test: Optional[Ne16Test] = None):
    if test is None:
        test = Ne16Test.load(args.test_dir)
    Ne16TestHeaderGenerator().generate(args.test_dir, test)


def test_gen(args):
    if args.conf.endswith(".toml"):
        test_conf_dict = toml.load(args.conf)
    elif args.conf.endswith(".json"):
        with open(args.conf, "r") as fp:
            test_conf_dict = json.load(fp)
    else:
        print(
            f"ERROR: Unsupported file type for {args.conf} configuration file. Supported file formats: .json and .toml."
        )
        exit(-1)

    test_conf = Ne16TestConf.model_validate(test_conf_dict)
    test = Ne16TestGenerator.from_conf(test_conf)
    if not args.skip_save:
        test.save(args.test_dir)
    if args.headers:
        headers_gen(args, test)


def _regen(path: Union[str, os.PathLike], regen_tensors: Set[str]) -> None:
    test = Ne16Test.load(path)
    test = Ne16TestGenerator.regenerate(test, regen_tensors)
    test.save(path)


def _regen_recursive(path: Union[str, os.PathLike], regen_tensors: Set[str]) -> None:
    if Ne16Test.is_test_dir(path):
        _regen(path, regen_tensors)
        return

    for dirpath, _, _ in os.walk(path):
        _regen_recursive(dirpath, regen_tensors)


def test_regen(args):
    regen_tensors = set(args.tensors + ["output"])

    for test_dir in args.test_dirs:
        if args.recurse:
            _regen_recursive(test_dir, regen_tensors)
        else:
            _regen(test_dir, regen_tensors)


parser = argparse.ArgumentParser(
    description="Utility script to generate tests and header files."
)

subparsers = parser.add_subparsers()

parser_header = subparsers.add_parser(
    "headers", description="Generate headers for a single test."
)
parser_header.add_argument(
    "-t",
    "--test-dir",
    type=str,
    dest="test_dir",
    required=True,
    help="Path to the test." "basename.",
)
parser_header.set_defaults(func=headers_gen)

parser_test = subparsers.add_parser(
    "test", description="Generate a test from a configuration."
)
parser_test.add_argument(
    "-c",
    "--conf",
    type=str,
    default="conf.toml",
    required=True,
    help="Path to the configuration file.",
)
parser_test.add_argument(
    "-t",
    "--test-dir",
    type=str,
    dest="test_dir",
    required=True,
    help="Path to the test. " "basename.",
)
parser_test.add_argument(
    "--headers", action="store_true", default=False, help="Generate headers."
)
parser_test.add_argument(
    "--skip-save",
    action="store_true",
    default=False,
    dest="skip_save",
    help="Skip saving the test.",
)
parser_test.set_defaults(func=test_gen)

parser_regen = subparsers.add_parser("regen", description="Regenerate test tensors.")
parser_regen.add_argument(
    "tensors",
    type=str,
    nargs="?",
    default=[],
    help="Tensors that should be regenerated. Output " "included by default.",
)
parser_regen.add_argument(
    "-t",
    "--test-dir",
    action="append",
    dest="test_dirs",
    required=True,
    help="Path to the test.",
)
parser_regen.add_argument(
    "-r",
    "--recursive",
    action="store_true",
    default=False,
    help="Recursively search for test directiories " "inside given test directories.",
)
parser_regen.set_defaults(func=test_regen)

args = parser.parse_args()

args.func(args)
