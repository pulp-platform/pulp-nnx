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

import numpy as np
import numpy.typing as npt
from TestClasses import IntegerType


class Ne16:
    ACCUMULATOR_TYPE = IntegerType(name="int32")

    _CIN_SUBTILE = 16

    @staticmethod
    def weight_unroll(
        weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False
    ) -> npt.NDArray[np.uint8]:
        """Unroll weight into expected memory format

        Expected weight shape is (Cout, Cin, H, W).
        The output shape is: (Cout, Cin_major, Bits, H x W, Cin_minor_bytes),
        where Cin_major is the ceil(Cin / CIN_SUBTILE) and Cin_minor has to be padded with 0 to CIN_SUBTILE.
        """
        if depthwise:
            weight = weight.transpose(1, 0, 2, 3)  # Swap Cout and Cin

        Cout, Cin, H, W = weight.shape

        # Pad Cin to be divisible with CIN_SUBTILE
        if Cin % Ne16._CIN_SUBTILE != 0:
            Cin_pad = Ne16._CIN_SUBTILE - Cin % Ne16._CIN_SUBTILE
            weight = np.pad(
                weight,
                ((0, 0), (0, Cin_pad), (0, 0), (0, 0)),
                "constant",
                constant_values=0,
            )

        # Reshape into (Cout, Cin_major, Cin_minor, Flattened spatial, 1)
        # The 1 at the end is required by the unpacking
        Cin_major = int(np.ceil(Cin / Ne16._CIN_SUBTILE))
        Cin_minor = Ne16._CIN_SUBTILE
        weight = weight.reshape(Cout, Cin_major, Cin_minor, H * W, 1)

        # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
        # (Cout, Cin_major, Cin_minor, Flattened spatial, Bits)
        weight = np.unpackbits(weight, axis=-1, count=bits, bitorder="little")

        # Shuffle bits so that the final shape is:
        # (Cout, Cin_major, Bits, Flattened spatial, Cin_minor)
        weight = weight.transpose(0, 1, 4, 3, 2)

        # Prepare for packing
        # (Cout, Cin_major, Bits, Flattened spatial, Cin_minor_bytes, 8)
        Cin_minor_bytes = int(np.ceil(Cin_minor / 8))
        weight = np.stack(np.split(weight, Cin_minor_bytes, axis=-1), axis=-2)

        # Pack
        # (Cout, Cin_major, Bits, Flattened spatial, Cin_minor_bytes)
        weight = np.packbits(weight, axis=-1, bitorder="little")

        return weight.flatten()

    @staticmethod
    def weight_roll(weight: np.ndarray, bits: int, Cout: int, Cin: int, H: int, W: int):
        """Reverse of weight_roll"""
        Cin_major = int(np.ceil(Cin / Ne16._CIN_SUBTILE))
        Cin_minor = Ne16._CIN_SUBTILE
        Cin_minor_bytes = int(np.ceil(Cin_minor / 8))

        weight = weight.reshape(Cout, Cin_major, bits, H * W, Cin_minor_bytes, 1)
        weight = np.unpackbits(weight, axis=-1, count=bits, bitorder="little")
        weight = weight.reshape(Cout, Cin_major, bits, H * W, Cin_minor)
        weight = weight.transpose(0, 1, 4, 3, 2)
        weight = np.packbits(weight, axis=-1, bitorder="little")
        weight = weight.reshape(Cout, Cin_major * Cin_minor, H, W)
        weight = weight[:, :Cin, :, :]

        return weight
