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

from functools import cached_property
import re
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING
from pydantic import (
    BaseModel,
    model_serializer,
    model_validator,
    NonNegativeInt,
    PositiveInt,
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

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, str):
            return self.name == __value
        elif isinstance(__value, IntegerType):
            return self.name == __value.name
        else:
            return False

    @model_serializer
    def ser_model(self) -> str:
        return self.name

    if TYPE_CHECKING:
        # Ensure type checkers see the correct return type
        def model_dump(
            self,
            *,
            mode: Literal["json", "python"] | str = "python",
            include: Any = None,
            exclude: Any = None,
            by_alias: bool = False,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            round_trip: bool = False,
            warnings: bool = True,
        ) -> dict[str, Any]:
            ...
