from typing import Optional
import torch
import torch.nn.functional as F
from NnxTestClasses import NnxTestConf
from TestClasses import IntegerType


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
        conf: NnxTestConf,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        global_shift: torch.Tensor,
    ) -> torch.Tensor:
        # Scale accumulators are in 48bit, so keeping the data in 64bit
        tensor = scale * tensor
        assert tensor.dtype == torch.int64

        if conf.has_bias:
            assert bias is not None
            assert conf.bias_type is not None
            # Saturating cast to int32
            tensor = NeuralEngineFunctionalModel._cast(
                tensor, conf.bias_type, saturate=True
            ).type(torch.int32)

            tensor = tensor + bias
            tensor = NeuralEngineFunctionalModel._cast(
                tensor, conf.bias_type, saturate=False
            ).type(torch.int32)

        if conf.has_relu:
            tensor = F.relu(tensor)

        tensor = tensor >> global_shift

        # Saturate into out_type
        tensor = NeuralEngineFunctionalModel._cast(tensor, conf.out_type, saturate=True)

        return tensor

    def convolution(
        self,
        conf: NnxTestConf,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        global_shift: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
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

        # Accumulators are 32bit non-saturating.
        # Calculate in higher precision (int64)
        output = F.conv2d(
            input=input_padded,
            weight=weight,
            stride=(conf.stride.height, conf.stride.width),
            groups=conf.in_channel if conf.depthwise else 1,
        ).type(torch.int64)

        # Cast to accumulator type
        output = NeuralEngineFunctionalModel._cast(
            output, NeuralEngineFunctionalModel.ACCUMULATOR_TYPE, saturate=False
        ).type(torch.int32)

        if verbose:
            print("INTERMEDIATE RESULTS (pre-normalization/requant):")
            print(output)

        if conf.has_norm_quant:
            assert scale is not None
            assert global_shift is not None
            output = self._norm_quant(conf, output, scale, bias, global_shift)

        return output
