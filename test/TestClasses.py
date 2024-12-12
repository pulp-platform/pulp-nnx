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
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveInt,
    model_serializer,
    model_validator,
)


def implies(a: bool, b: bool):
    return (not a) or b


class KernelShape(BaseModel):
    height: PositiveInt
    width: PositiveInt


class Stride(BaseModel):
    height: PositiveInt
    width: PositiveInt


class Padding(BaseModel):
    top: NonNegativeInt
    bottom: NonNegativeInt
    left: NonNegativeInt
    right: NonNegativeInt


class IntegerType(BaseModel):
    name: str

    @model_validator(mode="before")  # type: ignore
    @classmethod
    def model_validate_before(cls, data: Any) -> Dict:
        if isinstance(data, str):
            type_name = data
        elif isinstance(data, dict) and "name" in data:
            type_name = data["name"]
        else:
            raise ValueError(f"TestIntegerType: Invalid data {data} provided.")
        match = re.fullmatch(r"(u?int)(\d*)", type_name)
        assert (
            match
        ), f"TestIntegerType: Invalid integer type format {data}. Format should be u?int\\d*"
        return {"name": type_name}

    @cached_property
    def _signed(self) -> bool:
        match = re.fullmatch(r"(u?int)(\d*)", self.name)
        assert match is not None
        return match.group(1) == "int"

    @cached_property
    def _bits(self) -> int:
        match = re.fullmatch(r"(u?int)(\d*)", self.name)
        assert match is not None
        return int(match.group(2))

    @cached_property
    def min(self):
        return -(2 ** (self._bits - 1)) if self._signed else 0

    @cached_property
    def max(self):
        return (2 ** (self._bits - 1)) - 1 if self._signed else (2**self._bits) - 1

    def ctype(self) -> Optional[str]:
        if self._bits in [8, 16, 32, 64]:
            return f"{self}_t"
        else:
            return None

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, IntegerType):
            return self.name == other.name
        else:
            return False

    @model_serializer
    def ser_model(self) -> str:
        return self.name
