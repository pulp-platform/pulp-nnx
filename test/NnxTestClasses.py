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

from __future__ import annotations
from typing import Callable, Union, Optional, Set, Tuple, Type
import torch
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
import os
from HeaderWriter import HeaderWriter
from TestClasses import IntegerType, Stride, Padding, KernelShape, implies
from pydantic import BaseModel, PositiveInt


class NnxTestConf(BaseModel):
    in_height: PositiveInt
    in_width: PositiveInt
    in_channel: PositiveInt
    out_channel: PositiveInt
    padding: Padding
    kernel_shape: KernelShape
    depthwise: bool
    stride: Stride
    in_type: IntegerType
    out_type: IntegerType
    weight_type: IntegerType
    scale_type: Optional[IntegerType] = None
    bias_type: Optional[IntegerType] = None
    has_norm_quant: bool
    has_bias: bool
    has_relu: bool


class NnxTest:
    _CONF_NAME = "conf.json"
    _INPUT_NAME = "input.pt"
    _OUTPUT_NAME = "output.pt"
    _WEIGHT_NAME = "weight.pt"
    _SCALE_NAME = "scale.pt"
    _BIAS_NAME = "bias.pt"
    _GLOBAL_SHIFT_NAME = "global_shift.pt"

    def __init__(
        self,
        conf: NnxTestConf,
        input: Optional[torch.Tensor],
        output: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_shift: Optional[torch.Tensor] = torch.Tensor([0]),
    ) -> None:
        self.conf = conf
        self.input = input
        self.output = output
        self.weight = weight
        self.scale = scale
        self.bias = bias
        self.global_shift = global_shift

    def is_valid(self) -> bool:
        return all(
            [
                self.input is not None,
                self.output is not None,
                self.weight is not None,
                implies(self.conf.has_norm_quant, self.scale is not None),
                implies(self.conf.has_bias, self.bias is not None),
                implies(self.conf.has_norm_quant, self.global_shift is not None),
            ]
        )

    def save_conf(self, path: Union[str, os.PathLike]) -> None:
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, NnxTest._CONF_NAME), "w") as fp:
            fp.write(self.conf.model_dump_json(indent=4))

    def save_data(self, path: Union[str, os.PathLike]) -> None:
        os.makedirs(path, exist_ok=True)

        torch.save(self.input, os.path.join(path, NnxTest._INPUT_NAME))
        torch.save(self.output, os.path.join(path, NnxTest._OUTPUT_NAME))
        torch.save(self.weight, os.path.join(path, NnxTest._WEIGHT_NAME))
        if self.scale is not None:
            torch.save(self.scale, os.path.join(path, NnxTest._SCALE_NAME))
        if self.bias is not None:
            torch.save(self.bias, os.path.join(path, NnxTest._BIAS_NAME))
        if self.global_shift is not None:
            torch.save(
                self.global_shift, os.path.join(path, NnxTest._GLOBAL_SHIFT_NAME)
            )

    def save(self, path: Union[str, os.PathLike]) -> None:
        self.save_conf(path)
        self.save_data(path)

    @staticmethod
    def is_test_dir(path: Union[str, os.PathLike]) -> bool:
        fileset = set(os.listdir(path))
        required_fileset = set([NnxTest._CONF_NAME])
        return required_fileset.issubset(fileset)

    @classmethod
    def load(cls, confCls: Type[NnxTestConf], path: Union[str, os.PathLike]) -> NnxTest:
        assert NnxTest.is_test_dir(
            path
        ), f"ERROR: Test {path} does not contain the necessary files."

        with open(os.path.join(path, NnxTest._CONF_NAME), "r") as fp:
            conf = confCls.model_validate_json(fp.read())

        def load_if_exist(filename: str) -> Optional[torch.Tensor]:
            filepath = os.path.join(path, filename)
            return torch.load(filepath) if os.path.isfile(filepath) else None

        input = load_if_exist(NnxTest._INPUT_NAME)
        output = load_if_exist(NnxTest._OUTPUT_NAME)
        weight = load_if_exist(NnxTest._WEIGHT_NAME)
        scale = load_if_exist(NnxTest._SCALE_NAME)
        bias = load_if_exist(NnxTest._BIAS_NAME)
        global_shift = load_if_exist(NnxTest._GLOBAL_SHIFT_NAME)

        return cls(conf, input, output, weight, scale, bias, global_shift)


class NnxTestGenerator:
    _DEFAULT_SEED = 0

    @staticmethod
    def _global_shift(
        tensor: torch.Tensor, out_type: IntegerType, has_relu: bool
    ) -> torch.Tensor:
        if has_relu:
            # only adjust positive values
            tensor = tensor[tensor > 0]

        s = tensor.type(torch.float64).std()
        target_s = 2 ** (out_type._bits - 1)
        global_shift = torch.ceil(torch.log2(s / target_s)).type(torch.int32)

        return global_shift

    @staticmethod
    def _random_data(_type: IntegerType, shape: Tuple[int, int, int, int]):
        return torch.randint(_type.min, _type.max, size=shape)

    @staticmethod
    def _cast(
        tensor: torch.Tensor, _type: IntegerType, saturate: bool = False
    ) -> torch.Tensor:
        if saturate:
            return tensor.clamp(_type.min, _type.max)
        else:
            return tensor & ((1 << _type._bits) - 1)

    @staticmethod
    def from_conf(
        conf: NnxTestConf,
        accumulator_type: IntegerType,
        input: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_shift: Optional[torch.Tensor] = None,
    ) -> NnxTest:
        torch.manual_seed(NnxTestGenerator._DEFAULT_SEED)

        if input is None:
            input = NnxTestGenerator._random_data(
                _type=conf.in_type,
                shape=(1, conf.in_channel, conf.in_height, conf.in_width),
            )

        input_padded = F.pad(
            input,
            (
                conf.padding.left,
                conf.padding.right,
                conf.padding.top,
                conf.padding.bottom,
            ),
            "constant",
            0,
        )

        if weight is None:
            weight = NnxTestGenerator._random_data(
                _type=conf.weight_type,
                shape=(
                    conf.out_channel,
                    1 if conf.depthwise else conf.in_channel,
                    conf.kernel_shape.height,
                    conf.kernel_shape.width,
                ),
            )

        # Accumulators are 32bit non-saturating.
        # Calculate in higher precision (int64)
        output = F.conv2d(
            input=input_padded,
            weight=weight,
            stride=(conf.stride.height, conf.stride.width),
            groups=conf.in_channel if conf.depthwise else 1,
        ).type(torch.int64)
        # Use only the lower 32bits
        output = NnxTestGenerator._cast(output, accumulator_type, saturate=False).type(
            torch.int32
        )

        if conf.has_norm_quant:
            if scale is None:
                assert conf.scale_type is not None
                scale = NnxTestGenerator._random_data(
                    conf.scale_type, shape=(1, conf.out_channel, 1, 1)
                )
            # Scale accumulators are in 48bit, so keeping the data in 64bit
            output = scale * output
            assert output.dtype == torch.int64

            if conf.has_bias:
                # Saturating cast to int32
                assert conf.bias_type is not None
                output = NnxTestGenerator._cast(
                    output, conf.bias_type, saturate=True
                ).type(torch.int32)

                if bias is None:
                    bias = NnxTestGenerator._random_data(
                        conf.bias_type, shape=(1, conf.out_channel, 1, 1)
                    ).type(torch.int32)
                output = output + bias
                output = NnxTestGenerator._cast(
                    output, conf.bias_type, saturate=False
                ).type(torch.int32)

            if conf.has_relu:
                output = F.relu(output)

            if global_shift is None:
                global_shift = NnxTestGenerator._global_shift(
                    output, conf.out_type, conf.has_relu
                )
            output = output >> global_shift

            # Saturate into out_type
            output = NnxTestGenerator._cast(output, conf.out_type, saturate=True)

        return NnxTest(
            conf=conf,
            input=input,
            output=output,
            weight=weight,
            scale=scale,
            bias=bias,
            global_shift=global_shift,
        )

    @staticmethod
    def regenerate(test: NnxTest, regen_tensors: Set[str]) -> NnxTest:
        test_tensors = set(["input", "output", "weight", "scale", "bias"])
        load_tensors = test_tensors - regen_tensors
        kwargs = {tensor: getattr(test, tensor) for tensor in load_tensors}
        return NnxTestGenerator.from_conf(test.conf, **kwargs)


class NnxTestHeaderGenerator:
    DEFAULT_HEADERS_DIR = "app/gen"

    def __init__(
        self,
        weight_unroll: Callable[
            [npt.NDArray[np.uint8], int, bool], npt.NDArray[np.uint8]
        ],
        headers_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        if headers_dir is None:
            headers_dir = NnxTestHeaderGenerator.DEFAULT_HEADERS_DIR
        self.header_writer = HeaderWriter(headers_dir)
        # function that takes the weights in CoutCinK format, bitwidth, and a depthwise flag,
        # and returns a numpy array of dtype=np.uint8 of data in a layout correct for the accelerator
        self.weight_unroll = weight_unroll

    def generate(self, test_name: str, test: NnxTest):
        assert test.input is not None and test.output is not None
        _, in_channel, in_height, in_width = test.input.shape
        _, out_channel, out_height, out_width = test.output.shape

        # Render input
        in_ctype = test.conf.in_type.ctype()
        in_signed = test.conf.in_type._signed
        in_data = test.input.permute(0, 2, 3, 1).ravel()
        self.header_writer.generate_vector_files(
            "input", _type=in_ctype, size=in_data.numel(), init=in_data
        )

        # Render output
        out_ctype = test.conf.out_type.ctype()
        out_data_golden = test.output.permute(0, 2, 3, 1).ravel()
        self.header_writer.generate_vector_files(
            "output",
            _type=out_ctype,
            size=out_data_golden.numel(),
            golden=out_data_golden,
        )

        # Render weights
        assert test.weight is not None
        weight_type = test.conf.weight_type
        weight_bits = weight_type._bits
        assert weight_bits > 1 and weight_bits <= 8
        weight_offset = -(2 ** (weight_bits - 1))
        weight_out_ch, weight_in_ch, weight_ks_h, weight_ks_w = test.weight.shape
        weight_data: np.ndarray = test.weight.numpy() - weight_offset
        weight_init = self.weight_unroll(
            weight_data.astype(np.uint8),
            weight_type._bits,
            test.conf.depthwise,
        )
        self.header_writer.generate_vector_files(
            "weight", _type="uint8_t", size=weight_init.size, init=weight_init
        )

        # Render scale
        if test.scale is not None:
            assert test.conf.scale_type is not None
            scale_ctype = test.conf.scale_type.ctype()
            self.header_writer.generate_vector_files(
                "scale",
                _type=scale_ctype,
                size=test.scale.numel(),
                init=test.scale.ravel(),
            )

        # Render bias
        if test.bias is not None:
            assert test.conf.bias_type is not None
            bias_ctype = test.conf.bias_type.ctype()
            self.header_writer.generate_vector_files(
                "bias", _type=bias_ctype, size=test.bias.numel(), init=test.bias.ravel()
            )

        global_shift = 0 if test.global_shift is None else int(test.global_shift.item())

        # Render layer conf
        self.header_writer.generate_defines_header(
            "layer_conf",
            {
                "test_name": test_name,
                "input": {
                    "height": in_height,
                    "width": in_width,
                    "channel": in_channel,
                    "signed": in_signed,
                    "bits": 8,
                },
                "output": {
                    "height": out_height,
                    "width": out_width,
                    "channel": out_channel,
                    "bits": 8,
                },
                "weight": {
                    "height": weight_ks_h,
                    "width": weight_ks_w,
                    "channel_in": weight_in_ch,
                    "channel_out": weight_out_ch,
                    "bits": weight_bits,
                    "offset": weight_offset,
                },
                "scale": {"bits": 8},
                "bias": {"bits": 32},
                "padding": {
                    "top": test.conf.padding.top,
                    "bottom": test.conf.padding.bottom,
                    "left": test.conf.padding.left,
                    "right": test.conf.padding.right,
                    "value": 0,
                },
                "stride": test.conf.stride.model_dump(),
                "groups": test.conf.in_channel if test.conf.depthwise else 1,
                "outshift": global_shift,
                "has_norm_quant": test.conf.has_norm_quant,
                "has_bias": test.conf.has_bias,
                "has_relu": test.conf.has_relu,
            },
        )
