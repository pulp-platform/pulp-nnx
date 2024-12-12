# Luka Macan <luka.macan@unibo.it>
# Arpan Suravi Prasad <prasadar@iis.ee.ethz.ch>
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

from HeaderWriter import HeaderWriter
from NnxTestClasses import NnxWeight, WmemLiteral


class NeurekaV2Weight(NnxWeight):
    _WEIGHT_BANDWIDTH = 288
    _CIN_SUBTILE = 32

    @staticmethod
    def encode(
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
            NeurekaV2Weight._CIN_SUBTILE
            if height == 3
            else NeurekaV2Weight._CIN_SUBTILE
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
        weight = weight.reshape(cout, cinMajor, cinSubtile, height * width, 1)

        # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
        # (cout, cinMajor, cinSubtile, Flattened spatial, Bits)
        weight = np.unpackbits(weight, axis=-1, count=bits, bitorder="little")

        # Shuffle bits so that the final shape is:
        # (cout, cinMajor, Bits, Flattened spatial, cinSubtile)
        weight = weight.transpose(0, 1, 4, 3, 2)

        # Pack bits into bytes
        # (-1, 8)
        weight = weight.reshape(-1, 8)
        # (-1, 1)
        weight = np.packbits(weight, axis=-1, bitorder="little")

        # Flatten the weights
        # (-1, )
        return weight.flatten()

    @staticmethod
    def decode(
        weight: npt.NDArray[np.uint8],
        bits: int,
        cout: int,
        cin: int,
        height: int,
        width: int,
    ) -> npt.NDArray[np.uint8]:
        """Reverse of encode"""
        cinSubtile = (
            NeurekaV2Weight._CIN_SUBTILE
            if height == 3
            else NeurekaV2Weight._CIN_SUBTILE
        )
        cinMajor = int(np.ceil(cin / cinSubtile))
        cinMinor = cinSubtile
        weightBandwidthBytes = int(np.ceil(NeurekaV2Weight._WEIGHT_BANDWIDTH / 8))

        weight = weight.reshape(-1, 1)
        weight = np.unpackbits(weight, axis=-1, count=8, bitorder="little")

        if height == 3 and width == 3:
            weight = weight.reshape(-1, NeurekaV2Weight._WEIGHT_BANDWIDTH)
            weight = weight[:, : height * width * cinMinor]

        weight = weight.reshape(
            cout, cinMajor, bits, height * width, cinMinor
        ).transpose(0, 1, 4, 3, 2)
        weight = np.packbits(weight, axis=-1, bitorder="little")
        weight = weight.reshape(cout, cinMajor * cinMinor, height, width)
        weight = weight[:, :cin, :, :]

        return weight

    @staticmethod
    def source_generate(
        wmem: WmemLiteral, init: npt.NDArray[np.uint8], header_writer: HeaderWriter
    ) -> None:
        if wmem == "sram":
            section = '__attribute__((section(".weightmem_sram")))'
        elif wmem == "mram":
            section = '__attribute__((section(".weightmem_mram")))'
        else:
            section = "PI_L1"

        header_writer.generate_vector_files(
            "weight",
            _type="uint8_t",
            size=init.size,
            section=section,
        )

        header_writer.generate_vector_files(
            "weight_l2",
            _type="uint8_t",
            size=init.size,
            init=init,
            section="PI_L2",
        )
