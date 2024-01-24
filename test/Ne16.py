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
    _LINEAR_CIN_TILE = 256

    @staticmethod
    def weight_unroll(
        weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False
    ) -> npt.NDArray[np.uint8]:
        """Unroll weight into expected memory format

        Expected weight shape is (cout, cin, height, width).
        The output shape is: (cout, cinMajor, Bits, height x width, cinMinorBytes),
        where cinMajor is the ceil(cin / CIN_SUBTILE) and cinMinor has to be padded with 0 to CIN_SUBTILE.
        """
        if depthwise:
            weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

        cout, cin, height, width = weight.shape

        # Pad cin to be divisible with CIN_SUBTILE
        if cin % Ne16._CIN_SUBTILE != 0:
            cinPad = Ne16._CIN_SUBTILE - cin % Ne16._CIN_SUBTILE
            weight = np.pad(
                weight,
                ((0, 0), (0, cinPad), (0, 0), (0, 0)),
                "constant",
                constant_values=0,
            )
            cin = cin + cinPad

        # Reshape into (cout, cinMajor, cinMinor, flattened spatial, 1)
        # The 1 at the end is required by the unpacking
        cinMajor = cin // Ne16._CIN_SUBTILE
        cinMinor = Ne16._CIN_SUBTILE
        weight = weight.reshape(cout, cinMajor, cinMinor, height * width, 1)

        # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
        # (cout, cinMajor, cinMinor, flattened spatial, Bits)
        weight = np.unpackbits(weight, axis=-1, count=bits, bitorder="little")

        # Shuffle bits so that the final shape is:
        # (cout, cinMajor, Bits, flattened spatial, cinMinor)
        weight = weight.transpose(0, 1, 4, 3, 2)

        # Prepare for packing
        # (cout, cinMajor, Bits, flattened spatial, cinMinorBytes, 8)
        cinMinorBytes = int(np.ceil(cinMinor / 8))
        weight = np.stack(np.split(weight, cinMinorBytes, axis=-1), axis=-2)

        # Pack
        # (cout, cinMajor, Bits, flattened spatial, cinMinorBytes)
        weight = np.packbits(weight, axis=-1, bitorder="little")

        return weight.flatten()

    @staticmethod
    def weight_roll(
        weight: npt.NDArray[np.uint8],
        bits: int,
        cout: int,
        cin: int,
        height: int,
        width: int,
    ) -> npt.NDArray[np.uint8]:
        """Reverse of weight_roll"""
        cinMajor = int(np.ceil(cin / Ne16._CIN_SUBTILE))
        cinMinor = Ne16._CIN_SUBTILE
        cinMinorBytes = int(np.ceil(cinMinor / 8))

        weight = weight.reshape(cout, cinMajor, bits, height * width, cinMinorBytes, 1)
        weight = np.unpackbits(weight, axis=-1, count=8, bitorder="little")
        weight = weight.reshape(cout, cinMajor, bits, height * width, cinMinor)
        weight = weight.transpose(0, 1, 4, 3, 2)
        weight = np.packbits(weight, axis=-1, bitorder="little")
        weight = weight.reshape(cout, cinMajor * cinMinor, height, width)
        weight = weight[:, :cin, :, :]

        return weight
