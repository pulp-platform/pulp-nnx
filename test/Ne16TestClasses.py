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

from typing import List, Union, Sequence, Optional, Set
import torch
import numpy as np
import torch.nn.functional as F
import os
from Ne16 import Ne16
from HeaderWriter import HeaderWriter
from TestClasses import implies, KernelShape, Padding, Stride, IntegerType
from pydantic import BaseModel, field_validator, model_validator, PositiveInt


class Ne16TestConf(BaseModel):
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

    @field_validator("kernel_shape")
    @classmethod
    def check_valid_kernel_shape(cls, v: KernelShape) -> KernelShape:
        assert v == KernelShape(height=1, width=1) or v == KernelShape(
            height=3, width=3
        ), f"Unsupported kernel shape {v}. Supported 1x1 and 3x3."
        return v

    @field_validator("stride")
    @classmethod
    def check_valid_stride(cls, v: Stride) -> Stride:
        assert v == Stride(height=1, width=1) or v == Stride(
            height=2, width=2
        ), f"Unsupported stride {v}. Supported 1x1 and 2x2."
        return v

    @staticmethod
    def _check_type(
        name: str, _type: IntegerType, allowed_types: List[Union[IntegerType, str]]
    ) -> None:
        assert (
            _type in allowed_types
        ), f"Unsupported {name} type {_type}. Supported types: {allowed_types}"

    @field_validator("in_type")
    @classmethod
    def check_valid_in_type(cls, v: IntegerType) -> IntegerType:
        Ne16TestConf._check_type("in_type", v, ["uint8"])
        return v

    @field_validator("out_type")
    @classmethod
    def check_valid_out_type(cls, v: IntegerType) -> IntegerType:
        Ne16TestConf._check_type("out_type", v, ["uint8", "int8"])
        return v

    @field_validator("weight_type")
    @classmethod
    def check_valid_weight_type(cls, v: IntegerType) -> IntegerType:
        Ne16TestConf._check_type("weight_type", v, ["int8"])
        return v

    @field_validator("scale_type")
    @classmethod
    def check_valid_scale_type(cls, v: Optional[IntegerType]) -> Optional[IntegerType]:
        if v is not None:
            Ne16TestConf._check_type("scale_type", v, ["uint8", "uint32"])
        return v

    @field_validator("bias_type")
    @classmethod
    def check_valid_bias_type(cls, v: Optional[IntegerType]) -> Optional[IntegerType]:
        if v is not None:
            Ne16TestConf._check_type("bias_type", v, ["int32"])
        return v

    @model_validator(mode="after")
    def check_valid_out_channel_with_stride_2x2(self) -> "Ne16TestConf":
        assert implies(
            self.stride == Stride(height=2, width=2), self.out_channel % 2 == 0
        ), f"With stride 2x2 supported only even output channel sizes. Given output channel {self.out_channel}"
        return self

    @model_validator(mode="after")
    def check_valid_depthwise(self) -> "Ne16TestConf":
        assert implies(
            self.depthwise, self.kernel_shape == KernelShape(height=3, width=3)
        ), f"Depthwise supported only on 3x3 kernel shape. Given kernel shape {self.kernel_shape}."
        assert implies(self.depthwise, self.in_channel == self.out_channel), (
            f"Input and output channel should be the same in a depthwise layer. "
            f"input channel: {self.in_channel}, output channel: {self.out_channel}"
        )
        return self

    @model_validator(mode="after")
    def check_valid_padding_with_kernel_shape_1x1(self) -> "Ne16TestConf":
        assert implies(
            self.kernel_shape == KernelShape(height=1, width=1),
            self.padding == Padding(top=0, bottom=0, left=0, right=0),
        ), f"No padding on 1x1 kernel. Given padding {self.padding}"
        return self

    @field_validator("has_norm_quant")
    @classmethod
    def check_valid_has_norm_quant(cls, v: bool) -> bool:
        assert v == True, f"Untested without has_norm_quant."
        return v

    @model_validator(mode="after")
    def check_valid_norm_quant_types_when_has_norm_qunat(self) -> "Ne16TestConf":
        if self.has_norm_quant:
            assert self.scale_type is not None, "Scale type was not provided."
            if self.has_bias:
                assert self.bias_type is not None, "Bias type was not provided."
        return self

    @model_validator(mode="after")
    def check_valid_out_type_with_flags(self) -> "Ne16TestConf":
        assert implies(
            not self.has_norm_quant, self.out_type == Ne16.ACCUMULATOR_TYPE
        ), (
            f"Without quantization, the output type has to be equal to the "
            f"accumulator type {Ne16.ACCUMULATOR_TYPE}. Given output type {self.out_type}"
        )
        assert implies(
            self.has_norm_quant,
            (self.has_relu and not self.out_type._signed)
            or (not self.has_relu and self.out_type._signed),
        ), (
            f"Output type has to be unsigned when there is relu, otherwise signed. "
            f"Given output type {self.out_type} and has_relu {self.has_relu}"
        )
        return self


class Ne16Test:
    _CONF_NAME = "conf.json"
    _INPUT_NAME = "input.pt"
    _OUTPUT_NAME = "output.pt"
    _WEIGHT_NAME = "weight.pt"
    _SCALE_NAME = "scale.pt"
    _BIAS_NAME = "bias.pt"
    _GLOBAL_SHIFT_NAME = "global_shift.pt"

    def __init__(
        self,
        conf: Ne16TestConf,
        input: torch.Tensor,
        output: torch.Tensor,
        weight: torch.Tensor,
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

    def save(self, path: Union[str, os.PathLike]) -> None:
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, Ne16Test._CONF_NAME), "w") as fp:
            fp.write(self.conf.model_dump_json(indent=4))

        torch.save(self.input, os.path.join(path, Ne16Test._INPUT_NAME))
        torch.save(self.output, os.path.join(path, Ne16Test._OUTPUT_NAME))
        torch.save(self.weight, os.path.join(path, Ne16Test._WEIGHT_NAME))
        if self.scale is not None:
            torch.save(self.scale, os.path.join(path, Ne16Test._SCALE_NAME))
        if self.bias is not None:
            torch.save(self.bias, os.path.join(path, Ne16Test._BIAS_NAME))
        if self.global_shift is not None:
            torch.save(
                self.global_shift, os.path.join(path, Ne16Test._GLOBAL_SHIFT_NAME)
            )

    @staticmethod
    def is_test_dir(path: Union[str, os.PathLike]) -> bool:
        fileset = set(os.listdir(path))
        required_fileset = set(
            [
                Ne16Test._CONF_NAME,
                Ne16Test._INPUT_NAME,
                Ne16Test._OUTPUT_NAME,
                Ne16Test._WEIGHT_NAME,
            ]
        )
        return required_fileset.issubset(fileset)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Ne16Test":
        assert Ne16Test.is_test_dir(
            path
        ), f"ERROR: Test {path} does not contain the necessary files."

        with open(os.path.join(path, Ne16Test._CONF_NAME), "r") as fp:
            conf = Ne16TestConf.model_validate_json(fp.read())

        input = torch.load(os.path.join(path, Ne16Test._INPUT_NAME))
        output = torch.load(os.path.join(path, Ne16Test._OUTPUT_NAME))
        weight = torch.load(os.path.join(path, Ne16Test._WEIGHT_NAME))
        if os.path.isfile(os.path.join(path, Ne16Test._SCALE_NAME)):
            scale = torch.load(os.path.join(path, Ne16Test._SCALE_NAME))
        else:
            scale = None
        if os.path.isfile(os.path.join(path, Ne16Test._BIAS_NAME)):
            bias = torch.load(os.path.join(path, Ne16Test._BIAS_NAME))
        else:
            bias = None
        if os.path.isfile(os.path.join(path, Ne16Test._GLOBAL_SHIFT_NAME)):
            global_shift = torch.load(os.path.join(path, Ne16Test._GLOBAL_SHIFT_NAME))
        else:
            global_shift = None

        return cls(conf, input, output, weight, scale, bias, global_shift)


class Ne16TestGenerator:
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
    def _random_data(_type: IntegerType, shape: Sequence[int]):
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
        conf: Ne16TestConf,
        input: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_shift: Optional[torch.Tensor] = None,
    ) -> Ne16Test:
        if input is None:
            input = Ne16TestGenerator._random_data(
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
            weight = Ne16TestGenerator._random_data(
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
        output = Ne16TestGenerator._cast(
            output, Ne16.ACCUMULATOR_TYPE, saturate=False
        ).type(torch.int32)

        if conf.has_norm_quant:
            if scale is None:
                assert conf.scale_type is not None
                scale = Ne16TestGenerator._random_data(
                    conf.scale_type, shape=(1, conf.out_channel, 1, 1)
                )
            # Scale accumulators are in 48bit, so keeping the data in 64bit
            output = scale * output
            assert output.dtype == torch.int64

            if conf.has_bias:
                # Saturating cast to int32
                assert conf.bias_type is not None
                output = Ne16TestGenerator._cast(
                    output, conf.bias_type, saturate=True
                ).type(torch.int32)

                if bias is None:
                    bias = Ne16TestGenerator._random_data(
                        conf.bias_type, shape=(1, conf.out_channel, 1, 1)
                    ).type(torch.int32)
                output = output + bias
                output = Ne16TestGenerator._cast(
                    output, conf.bias_type, saturate=False
                ).type(torch.int32)

            if conf.has_relu:
                output = F.relu(output)

            if global_shift is None:
                global_shift = Ne16TestGenerator._global_shift(
                    output, conf.out_type, conf.has_relu
                )
            output = output >> global_shift

            # Saturate into out_type
            output = Ne16TestGenerator._cast(output, conf.out_type, saturate=True)

        return Ne16Test(
            conf=conf,
            input=input,
            output=output,
            weight=weight,
            scale=scale,
            bias=bias,
            global_shift=global_shift,
        )

    @staticmethod
    def regenerate(test: Ne16Test, regen_tensors: Set[str]) -> Ne16Test:
        test_tensors = set(["input", "output", "weight", "scale", "bias"])
        load_tensors = test_tensors - regen_tensors
        kwargs = {tensor: getattr(test, tensor) for tensor in load_tensors}
        return Ne16TestGenerator.from_conf(test.conf, **kwargs)


class Ne16TestHeaderGenerator:
    DEFAULT_HEADERS_DIR = "app/gen_inc"

    def __init__(self, headers_dir: Optional[Union[str, os.PathLike]] = None):
        if headers_dir is None:
            headers_dir = Ne16TestHeaderGenerator.DEFAULT_HEADERS_DIR
        os.makedirs(headers_dir, exist_ok=True)
        self.header_writer = HeaderWriter(headers_dir)

    def generate(self, test_name: str, test: Ne16Test):
        _, in_channel, in_height, in_width = test.input.shape
        _, out_channel, out_height, out_width = test.output.shape

        # Render input
        in_ctype = test.conf.in_type.ctype()
        in_data = test.input.permute(0, 2, 3, 1).ravel()
        self.header_writer.generate_vector_header(
            "input", _type=in_ctype, size=in_data.numel(), init=in_data
        )

        # Render output
        out_ctype = test.conf.out_type.ctype()
        out_data_golden = test.output.permute(0, 2, 3, 1).ravel()
        self.header_writer.generate_vector_header(
            "output",
            _type=out_ctype,
            size=out_data_golden.numel(),
            golden=out_data_golden,
        )

        # Render weights
        weight_type = test.conf.weight_type
        weight_bits = weight_type._bits
        assert weight_bits > 1 and weight_bits <= 8
        weight_offset = -(2 ** (weight_bits - 1))
        weight_out_ch, weight_in_ch, weight_ks_h, weight_ks_w = test.weight.shape
        weight_data: np.ndarray = test.weight.numpy() - weight_offset
        weight_init = Ne16.weight_unroll(
            weight_data.astype(np.uint8),
            weight_type._bits,
            depthwise=test.conf.depthwise,
        )
        self.header_writer.generate_vector_header(
            "weight", _type="uint8_t", size=weight_init.size, init=weight_init
        )

        # Render scale
        if test.scale is not None:
            assert test.conf.scale_type is not None
            scale_ctype = test.conf.scale_type.ctype()
            self.header_writer.generate_vector_header(
                "scale",
                _type=scale_ctype,
                size=test.scale.numel(),
                init=test.scale.ravel(),
            )

        # Render bias
        if test.bias is not None:
            assert test.conf.bias_type is not None
            bias_ctype = test.conf.bias_type.ctype()
            self.header_writer.generate_vector_header(
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
