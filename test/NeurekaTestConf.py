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
from Neureka import Neureka
from typing import List, Union, Optional
from NnxTestClasses import NnxTestConf
from TestClasses import implies, KernelShape, Padding, Stride, IntegerType
from pydantic import field_validator, model_validator


class NeurekaTestConf(NnxTestConf):
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
        assert v == Stride(height=1, width=1), f"Unsupported stride {v}. Supported 1x1."
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
        NeurekaTestConf._check_type("in_type", v, ["uint8"])
        return v

    @field_validator("out_type")
    @classmethod
    def check_valid_out_type(cls, v: IntegerType) -> IntegerType:
        NeurekaTestConf._check_type("out_type", v, ["uint8", "int8"])
        return v

    @field_validator("weight_type")
    @classmethod
    def check_valid_weight_type(cls, v: IntegerType) -> IntegerType:
        NeurekaTestConf._check_type("weight_type", v, ["int8"])
        return v

    @field_validator("scale_type")
    @classmethod
    def check_valid_scale_type(cls, v: Optional[IntegerType]) -> Optional[IntegerType]:
        if v is not None:
            NeurekaTestConf._check_type("scale_type", v, ["uint8", "uint32"])
        return v

    @field_validator("bias_type")
    @classmethod
    def check_valid_bias_type(cls, v: Optional[IntegerType]) -> Optional[IntegerType]:
        if v is not None:
            NeurekaTestConf._check_type("bias_type", v, ["int32"])
        return v

    @model_validator(mode="after")  # type: ignore
    def check_valid_depthwise(self) -> NeurekaTestConf:
        assert implies(
            self.depthwise, self.kernel_shape == KernelShape(height=3, width=3)
        ), f"Depthwise supported only on 3x3 kernel shape. Given kernel shape {self.kernel_shape}."
        assert implies(self.depthwise, self.in_channel == self.out_channel), (
            f"Input and output channel should be the same in a depthwise layer. "
            f"input channel: {self.in_channel}, output channel: {self.out_channel}"
        )
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_padding_with_kernel_shape_1x1(self) -> NeurekaTestConf:
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

    @model_validator(mode="after")  # type: ignore
    def check_valid_norm_quant_types_when_has_norm_qunat(self) -> NeurekaTestConf:
        if self.has_norm_quant:
            assert self.scale_type is not None, "Scale type was not provided."
            if self.has_bias:
                assert self.bias_type is not None, "Bias type was not provided."
        return self

    @model_validator(mode="after")  # type: ignore
    def check_valid_out_type_with_flags(self) -> NeurekaTestConf:
        assert implies(
            not self.has_norm_quant, self.out_type == Neureka.ACCUMULATOR_TYPE
        ), (
            f"Without quantization, the output type has to be equal to the "
            f"accumulator type {Neureka.ACCUMULATOR_TYPE}. Given output type {self.out_type}"
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
