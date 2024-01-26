from typing import Optional

import torch
import torch.nn.functional as F

from TestClasses import IntegerType, Padding, Stride


class NeuralEngineFunctionalModel:
    ACCUMULATOR_TYPE = IntegerType(name="int32")

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
    ) -> torch.Tensor:
        # Scale accumulators are in 48bit, so keeping the data in 64bit
        tensor = tensor * scale
        assert tensor.dtype == torch.int64

        if has_bias:
            assert bias is not None
            assert bias_type is not None
            # Saturating cast to int32
            tensor = NeuralEngineFunctionalModel._cast(
                tensor, bias_type, saturate=True
            ).type(torch.int32)

            tensor = tensor + bias
            tensor = NeuralEngineFunctionalModel._cast(
                tensor, bias_type, saturate=False
            ).type(torch.int32)

        if has_relu:
            tensor = F.relu(tensor)

        tensor = tensor >> global_shift

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
            print(output)

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
            )

        return output
