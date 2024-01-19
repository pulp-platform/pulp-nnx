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
import argparse
import json
import toml
from typing import Optional, Type, Union, Set
from Ne16 import Ne16
from Ne16TestConf import Ne16TestConf
from Neureka import Neureka
from NeurekaTestConf import NeurekaTestConf
from NnxTestClasses import (
    NnxTest,
    NnxTestConf,
    NnxTestGenerator,
    NnxTestHeaderGenerator,
)


def headers_gen(
    args,
    nnxCls: Union[Type[Ne16], Type[Neureka]],
    nnxTestConfCls: Type[NnxTestConf],
    test: Optional[NnxTest] = None,
):
    if test is None:
        test = NnxTest.load(nnxTestConfCls, args.test_dir)
    assert test is not None
    if not test.is_valid():
        test = NnxTestGenerator.from_conf(test.conf, nnxCls.ACCUMULATOR_TYPE)
    NnxTestHeaderGenerator(nnxCls.weight_unroll).generate(args.test_dir, test)


def print_tensors(test: NnxTest):
    print("INPUT TENSOR:")
    print(test.input)
    print("WEIGHT TENSOR:")
    print(test.weight)
    print("SCALE TENSOR:")
    print(test.scale)
    print("BIAS TENSOR:")
    print(test.bias)
    print("GLOBAL SHIFT TENSOR:")
    print(test.global_shift)
    print("EXPECTED OUTPUT TENSOR:")
    print(test.output)


def test_gen(
    args, nnxCls: Union[Type[Ne16], Type[Neureka]], nnxTestConfCls: Type[NnxTestConf]
):
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

    test_conf = nnxTestConfCls.model_validate(test_conf_dict)
    test = NnxTestGenerator.from_conf(
        test_conf, nnxCls.ACCUMULATOR_TYPE, verbose=args.print_tensors
    )
    if not args.skip_save:
        test.save(args.test_dir)
    if args.headers:
        headers_gen(args, nnxCls, nnxTestConfCls, test)
    if args.print_tensors:
        print_tensors(test)


def _regen(
    path: Union[str, os.PathLike],
    regen_tensors: Set[str],
    nnxTestConfCls: Type[NnxTestConf],
) -> None:
    test = NnxTest.load(nnxTestConfCls, path)
    test = NnxTestGenerator.regenerate(test, regen_tensors)
    test.save(path)


def _regen_recursive(
    path: Union[str, os.PathLike],
    regen_tensors: Set[str],
    nnxTestConfCls: Type[NnxTestConf],
) -> None:
    if NnxTest.is_test_dir(path):
        _regen(path, regen_tensors, nnxTestConfCls)
        return

    for dirpath, _, _ in os.walk(path):
        _regen_recursive(dirpath, regen_tensors, nnxTestConfCls)


def test_regen(
    args, nnxCls: Union[Type[Ne16], Type[Neureka]], nnxTestConfCls: Type[NnxTestConf]
):
    _ = nnxCls
    regen_tensors = set(args.tensors + ["output"])

    for test_dir in args.test_dirs:
        if args.recurse:
            _regen_recursive(test_dir, regen_tensors, nnxTestConfCls)
        else:
            _regen(test_dir, regen_tensors, nnxTestConfCls)


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-t",
        "--test-dir",
        type=str,
        dest="test_dir",
        required=True,
        help="Path to the test.",
    )

    parser.add_argument(
        "-a",
        "--accelerator",
        choices=["ne16", "neureka"],
        default="ne16",
        help="Choose an accelerator. Default: ne16",
    )


parser = argparse.ArgumentParser(
    description="Utility script to generate tests and header files."
)

subparsers = parser.add_subparsers()

parser_header = subparsers.add_parser(
    "headers", description="Generate headers for a single test."
)
add_common_arguments(parser_header)
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
    "--headers", action="store_true", default=False, help="Generate headers."
)
parser_test.add_argument(
    "--skip-save",
    action="store_true",
    default=False,
    dest="skip_save",
    help="Skip saving the test.",
)
parser_test.add_argument(
    "--print-tensors",
    action="store_true",
    default=False,
    dest="print_tensors",
    help="Print tensor values to stdout.",
)
add_common_arguments(parser_test)
parser_test.set_defaults(func=test_gen)

parser_regen = subparsers.add_parser("regen", description="Regenerate test tensors.")
parser_regen.add_argument(
    "tensors",
    type=str,
    nargs="?",
    default=[],
    help="Tensors that should be regenerated. Output included by default.",
)
parser_regen.add_argument(
    "-r",
    "--recursive",
    action="store_true",
    default=False,
    help="Recursively search for test directiories inside given test directories.",
)
add_common_arguments(parser_regen)
parser_regen.set_defaults(func=test_regen)

args = parser.parse_args()

if args.accelerator == "ne16":
    nnxCls = Ne16
    nnxTestConfCls = Ne16TestConf
elif args.accelerator == "neureka":
    nnxCls = Neureka
    nnxTestConfCls = NeurekaTestConf
else:
    assert False, f"Unsupported accelerator {args.accelerator}."

args.func(args, nnxCls, nnxTestConfCls)
