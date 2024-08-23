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

import os
from typing import Callable, Optional, Set, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, PositiveInt, field_validator, model_validator

from HeaderWriter import HeaderWriter
from NeuralEngineFunctionalModel import NeuralEngineFunctionalModel
from TestClasses import IntegerType, KernelShape, Padding, Stride, implies


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
    synthetic_weights: bool
    synthetic_inputs: bool

    @model_validator(mode="after")  # type: ignore
    def check_valid_depthwise_channels(self) -> NnxTestConf:
        assert implies(self.depthwise, self.in_channel == self.out_channel), (
            f"Input and output channel should be the same in a depthwise layer. "
            f"input channel: {self.in_channel}, output channel: {self.out_channel}"
        )
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_padding_with_kernel_shape_1x1(self) -> NnxTestConf:
        assert implies(
            self.kernel_shape == KernelShape(height=1, width=1),
            self.padding == Padding(top=0, bottom=0, left=0, right=0),
        ), f"No padding on 1x1 kernel. Given padding {self.padding}"
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_norm_quant_types_when_has_norm_qunat(self) -> NnxTestConf:
        if self.has_norm_quant:
            assert self.scale_type is not None, "Scale type was not provided."
            if self.has_bias:
                assert self.bias_type is not None, "Bias type was not provided."
        return self

    @model_validator(mode="after")  # type: ignore
    def check_has_relu_with_norm_quant(self) -> NnxTestConf:
        assert implies(self.has_relu, self.has_norm_quant), (
            f"Relu flag can only be enabled when norm_quant is enabled. "
            f"Given has_relu {self.has_relu} and has_norm_quant {self.has_norm_quant}"
        )
        return self

    @model_validator(mode="after")  # type: ignore
    def check_has_bias_with_norm_quant(self) -> NnxTestConf:
        assert implies(self.has_bias, self.has_norm_quant), (
            f"Bias flag can only be enabled when norm_quant is enabled. "
            f"Given has_bias {self.has_bias} and has_norm_quant {self.has_norm_quant}"
        )
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_out_type_with_relu(self) -> NnxTestConf:
        assert self.has_relu ^ self.out_type._signed, (
            f"Output type has to be unsigned when there is relu, otherwise signed. "
            f"Given output type {self.out_type} and has_relu {self.has_relu}"
        )
        return self


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
        synthetic_weights: Optional[bool] = False,
        synthetic_inputs: Optional[bool] = False,
    ) -> None:
        self.conf = conf
        self.input = input
        self.output = output
        self.weight = weight
        self.scale = scale
        self.bias = bias
        self.global_shift = global_shift
        self.synthetic_weights = synthetic_weights
        self.synthetic_inputs = synthetic_inputs

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
    _DEFAULT_WEIGHT_MEAN = 0.5 # as we use torch.floor(), this makes the generation unbiased
    _DEFAULT_WEIGHT_STDEV = 0.27
    _DEFAULT_SCALE_MAX_BIT_32BIT = 17
    _DEFAULT_SCALE_MAX_BIT_16BIT = 11
    _DEFAULT_SCALE_MAX_BIT_8BIT = 5
    _DEFAULT_BIAS_MAX_BIT = 18

    @staticmethod
    def _calculate_global_shift(
        tensor: torch.Tensor, out_type: IntegerType
    ) -> torch.Tensor:
        """Calculate global shift so that the output values are in the range of out_type"""
        s = tensor.type(torch.float64).std()
        target_s = 2 ** (out_type._bits - 1)
        shift = torch.ceil(torch.log2(s / target_s)).type(torch.int32)
        if shift < 1:
            return torch.zeros((1,)).type(torch.int32)
        else:
            return shift

    @staticmethod
    def _random_data(_type: IntegerType, shape: Tuple, extremes: Tuple = None):
        if extremes is None:
            return torch.randint(_type.min, _type.max, size=shape)
        else:
            return torch.randint(max(_type.min, extremes[0]), min(_type.max, extremes[1]), size=shape)

    @staticmethod
    def _random_data_normal(_type: IntegerType, shape: Tuple, mean: float64 = 0.5, std: float64=0.27):
        return torch.floor(torch.clip(torch.normal(mean, std, size=shape), _type.min, _type.max)).type(torch.int64)

    @staticmethod
    def from_conf(
        conf: NnxTestConf,
        input: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_shift: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> NnxTest:
        torch.manual_seed(NnxTestGenerator._DEFAULT_SEED)

        input_shape = (1, conf.in_channel, conf.in_height, conf.in_width)
        weight_shape = (
            conf.out_channel,
            1 if conf.depthwise else conf.in_channel,
            conf.kernel_shape.height,
            conf.kernel_shape.width,
        )
        scale_shape = (1, conf.out_channel, 1, 1)
        bias_shape = (1, conf.out_channel, 1, 1)

        if input is None:
            if conf.synthetic_inputs:
                inputs = torch.zeros((1, conf.in_channel, conf.in_height, conf.in_width), dtype=torch.int64)
                for i in range(conf.in_channel):
                    inputs[:, i,0,0] = i
            else:
                input = NnxTestGenerator._random_data(
                    _type=conf.in_type,
                    shape=input_shape,
                )

        if weight is None:
            if conf.synthetic_weights:
                weight = torch.zeros((conf.out_channel, 1 if conf.depthwise else conf.in_channel, conf.kernel_shape.height, conf.kernel_shape.width), dtype=torch.int64)
                for i in range(0, min(weight.shape[0], weight.shape[1])):
                    weight[i,i,0,0] = 1
            else:
                weight_mean = NnxTestGenerator._DEFAULT_WEIGHT_MEAN
                weight_std  = NnxTestGenerator._DEFAULT_WEIGHT_STDEV * (1<<(conf.weight_type._bits-1)-1)
                weight = NnxTestGenerator._random_data_normal(
                    mean = weight_mean,
                    std = weight_std,
                    _type=conf.weight_type,
                    shape=weight_shape,
                )

        if conf.has_norm_quant:
            if scale is None:
                assert conf.scale_type is not None
                # same limits as in old NE16 generator
                scale_extremes = (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_32BIT)-1) if conf.scale_type._bits == 32 else \
                                 (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_16BIT)-1) if conf.scale_type._bits == 16 else \
                                 (1, (1<<NnxTestGenerator._DEFAULT_SCALE_MAX_BIT_8BIT)-1)  if conf.scale_type._bits == 8  else (1, (1<<18)-1)
                scale = NnxTestGenerator._random_data(
                    conf.scale_type, shape=scale_shape, extremes=scale_extremes
                )
            if conf.has_bias and bias is None:
                assert conf.bias_type is not None
                # same limits as in old NE16 generator
                bias_extremes = (-(1<<NnxTestGenerator._DEFAULT_BIAS_MAX_BIT), (1<<NnxTestGenerator._DEFAULT_BIAS_MAX_BIT)-1)
                bias = NnxTestGenerator._random_data(
                    conf.bias_type, shape=bias_shape, extremes=bias_extremes
                ).type(torch.int32)
            if global_shift is None:
                global_shift = torch.Tensor([0]).type(torch.int32)
                conv_kwargs = {
                    **conf.__dict__,
                    "out_type": NeuralEngineFunctionalModel.ACCUMULATOR_TYPE,
                }
                output = NeuralEngineFunctionalModel().convolution(
                    input,
                    weight,
                    scale,
                    bias,
                    global_shift,
                    verbose=False,
                    **conv_kwargs,
                )
                global_shift = NnxTestGenerator._calculate_global_shift(
                    output, conf.out_type
                )

        output = NeuralEngineFunctionalModel().convolution(
            input, weight, scale, bias, global_shift, verbose=verbose, **conf.__dict__
        )

        return NnxTest(
            conf=conf,
            input=input,
            output=output,
            weight=weight,
            scale=scale,
            bias=bias,
            global_shift=global_shift,
            synthetic_inputs=conf.synthetic_inputs,
            synthetic_weights=conf.synthetic_weights,
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
        weightEncode: Callable[
            [npt.NDArray[np.uint8], int, bool], npt.NDArray[np.uint8]
        ],
        headers_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        if headers_dir is None:
            headers_dir = NnxTestHeaderGenerator.DEFAULT_HEADERS_DIR
        self.header_writer = HeaderWriter(headers_dir)
        # function that takes the weights in CoutCinK format, bitwidth, and a depthwise flag,
        # and returns a numpy array of dtype=np.uint8 of data in a layout correct for the accelerator
        self.weightEncode = weightEncode

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
        if test.synthetic_weights:
            weight_offset = 0
        else:
            weight_offset = -(2 ** (weight_bits - 1))
        weight_out_ch, weight_in_ch, weight_ks_h, weight_ks_w = test.weight.shape
        weight_data: np.ndarray = test.weight.numpy() - weight_offset
        weight_init = self.weightEncode(
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
                    "bits": test.conf.in_type._bits,
                },
                "output": {
                    "height": out_height,
                    "width": out_width,
                    "channel": out_channel,
                    "bits": test.conf.out_type._bits,
                },
                "weight": {
                    "height": weight_ks_h,
                    "width": weight_ks_w,
                    "channel_in": weight_in_ch,
                    "channel_out": weight_out_ch,
                    "bits": weight_bits,
                    "offset": weight_offset,
                },
                "scale": {
                    "bits": test.conf.scale_type._bits
                    if test.conf.scale_type is not None
                    else 0
                },
                "bias": {
                    "bits": test.conf.bias_type._bits
                    if test.conf.bias_type is not None
                    else 0
                },
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
