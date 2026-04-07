from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from TestClasses import IntegerType, Padding, Stride


class NeuralEngineFunctionalModel:
    ACCUMULATOR_TYPE = IntegerType(name="int32")

    @staticmethod
    def _tensor_to_hex(tensor):
        int_tensor = np.asarray(torch.floor(tensor).to(torch.int64))
        int_tensor[int_tensor < 0] = 0xffffffff + (int_tensor[int_tensor < 0]+1)
        hex_tensor = np.empty(int_tensor.shape, dtype=object)
        for idx in np.ndindex(int_tensor.shape):
            hex_tensor[idx] = hex(int_tensor[idx].item())
        return hex_tensor

    @staticmethod
    def _cast(
        tensor: torch.Tensor, _type: IntegerType, saturate: bool = False
    ) -> torch.Tensor:
        if saturate:
            return tensor.clamp(_type.min, _type.max)
        else:
            return tensor & ((1 << _type._bits) - 1)

    def _norm_quant(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        global_shift: torch.Tensor,
        out_type: IntegerType,
        bias_type: Optional[IntegerType],
        has_bias: bool,
        has_relu: bool,
        verbose: bool,
    ) -> torch.Tensor:
        # Scale accumulators are in 48bit, so keeping the data in 64bit
        tensor = tensor * scale
        assert tensor.dtype == torch.int64

        if verbose:
            print("INTERMEDIATE RESULTS (after scale):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(tensor))
            np.set_printoptions(threshold=current_threshold)

        if has_bias:
            assert bias is not None
            assert bias_type is not None

            tensor = NeuralEngineFunctionalModel._cast(
                tensor, bias_type, saturate=False
            ).type(torch.int32)

            tensor = tensor + bias

            tensor = NeuralEngineFunctionalModel._cast(
                tensor, bias_type, saturate=True
            ).type(torch.int32)

            if verbose:
                print("INTERMEDIATE RESULTS (after bias):")
                current_threshold = np.get_printoptions()['threshold']
                np.set_printoptions(threshold=np.inf)
                print(NeuralEngineFunctionalModel._tensor_to_hex(tensor))
                np.set_printoptions(threshold=current_threshold)

        if has_relu:
            tensor = F.relu(tensor)

        tensor = tensor >> global_shift

        if verbose:
            print("INTERMEDIATE RESULTS (after shift):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(tensor))
            np.set_printoptions(threshold=current_threshold)

        # Saturate into out_type
        tensor = NeuralEngineFunctionalModel._cast(tensor, out_type, saturate=True)

        return tensor

    def convolution(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        global_shift: Optional[torch.Tensor],
        padding: Padding,
        stride: Stride,
        depthwise: bool,
        out_type: IntegerType,
        bias_type: Optional[IntegerType],
        has_norm_quant: bool,
        has_bias: bool,
        has_relu: bool,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        _ = kwargs

        input_padded = F.pad(
            input,
            (
                padding.left,
                padding.right,
                padding.top,
                padding.bottom,
            ),
            "constant",
            0,
        )

        if verbose:
            print("INPUTS (padded):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(input_padded))
            print("WEIGHTS (padded):")
            print(NeuralEngineFunctionalModel._tensor_to_hex(weight))
            np.set_printoptions(threshold=current_threshold)

        # Accumulators are 32bit non-saturating.
        # Calculate in higher precision (int64)
        output = F.conv2d(
            input=input_padded,
            weight=weight,
            stride=(stride.height, stride.width),
            groups=weight.shape[0] if depthwise else 1,
        ).type(torch.int64)

        # Cast to accumulator type
        output = NeuralEngineFunctionalModel._cast(
            output, NeuralEngineFunctionalModel.ACCUMULATOR_TYPE, saturate=False
        ).type(torch.int32)

        if verbose:
            print("INTERMEDIATE RESULTS (pre-normalization/requant):")
            current_threshold = np.get_printoptions()['threshold']
            np.set_printoptions(threshold=np.inf)
            print(NeuralEngineFunctionalModel._tensor_to_hex(output))
            np.set_printoptions(threshold=current_threshold)

        if has_norm_quant:
            assert scale is not None
            assert global_shift is not None
            output = self._norm_quant(
                output,
                scale,
                bias,
                global_shift,
                out_type,
                bias_type,
                has_bias,
                has_relu,
                verbose,
            )

        return output
