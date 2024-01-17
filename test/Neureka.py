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


class Neureka:
    ACCUMULATOR_TYPE = IntegerType(name="int32")

    _WEIGHT_BANDWIDTH = 256
    _CIN_SUBTILE_1x1 = 32
    _CIN_SUBTILE_3x3 = 28

    @staticmethod
    def weight_unroll(
        weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False
    ) -> npt.NDArray[np.uint8]:
        """Unroll weight into expected memory format

        Expected weight shape is (cout, cin, H, W).
        The produced memory layout depends on the weight kernel shape:
          - 3x3: (cout, cinMajor, Bits, H x W x cinMinor_3x3 packed into Weight Bandwidth bits),
          - 1x1: (cout, cinMajor, Bits x H x W x cinMinor_1x1 packed into Weight Bandwidth bits),
        where cinMajor is the ceil(cin / cin subtile <mode>) and cinMinor has to be padded with 0 to cin subtile <mode>.
        """
        if depthwise:
            weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

        cout, cin, height, width = weight.shape
        cinSubtile = (
            Neureka._CIN_SUBTILE_3x3 if height == 3 else Neureka._CIN_SUBTILE_1x1
        )

        # Pad cin to be divisible with CIN_SUBTILE
        if cin % cinSubtile != 0:
            cinPad = cinSubtile - cin % cinSubtile
            weight = np.pad(
                weight,
                ((0, 0), (0, cinPad), (0, 0), (0, 0)),
                "constant",
                constant_values=0,
            )

        # Reshape into (cout, cinMajor, cinMinor, Flattened spatial, 1)
        # The 1 at the end is required by the unpacking
        cinMajor = int(np.ceil(cin / cinSubtile))
        cinMinor = cinSubtile
        weight = weight.reshape(cout, cinMajor, cinMinor, height * width, 1)

        # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
        # (cout, cinMajor, cinMinor, Flattened spatial, Bits)
        weight = np.unpackbits(weight, axis=-1, count=bits, bitorder="little")

        # Shuffle bits so that the final shape is:
        # (cout, cinMajor, Bits, Flattened spatial, cinMinor)
        weight = weight.transpose(0, 1, 4, 3, 2)

        # Pack dimensions to fit into weight bandwidth
        if height == 3 and width == 3:
            # (cout * cinMajor * Bits, H * W * cinMinor)
            weight = weight.reshape(-1, height * width * cinMinor)
        elif height == 1 and width == 1:
            # (cout * cinMajor, Bits * H * W * cinMinor)
            weight = weight.reshape(-1, bits * height * width * cinMinor)

        # Pad only the last dimension to weight bandwidth size
        # (-1, Weight Bandwidth)
        weight = np.pad(
            weight,
            ((0, 0), (0, Neureka._WEIGHT_BANDWIDTH - weight.shape[-1])),
            "constant",
            constant_values=0,
        )

        # Prepare for packing
        # (-1, Weight Bandwidth Bytes, 8)
        weightBandwidthBytes = int(np.ceil(Neureka._WEIGHT_BANDWIDTH / 8))
        weight = np.stack(np.split(weight, weightBandwidthBytes, axis=-1), axis=-2)

        # Pack bits
        # (-1, Weight Bandwidth Bytes)
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
        cinSubtile = (
            Neureka._CIN_SUBTILE_3x3 if height == 3 else Neureka._CIN_SUBTILE_1x1
        )
        cinMajor = int(np.ceil(cin / cinSubtile))
        cinMinor = cinSubtile
        weightBandwidthBytes = int(np.ceil(Neureka._WEIGHT_BANDWIDTH / 8))

        weight = weight.reshape(-1, weightBandwidthBytes, 1)
        weight = np.unpackbits(weight, axis=-1, count=8, bitorder="little")
        weight = weight.reshape(-1, Neureka._WEIGHT_BANDWIDTH)
        if height == 3 and width == 3:
            weight = weight[:, : height * width * cinMinor]
        elif height == 1 and width == 1:
            weight = weight[:, : bits * height * width * cinMinor]
        weight = weight.reshape(cout, cinMajor, bits, height * width, cinMinor)
        weight = weight.transpose(0, 1, 4, 3, 2)
        weight = np.packbits(weight, axis=-1, bitorder="little")
        weight = weight.reshape(cout, cinMajor * cinMinor, height, width)
        weight = weight[:, :cin, :, :]

        return weight
