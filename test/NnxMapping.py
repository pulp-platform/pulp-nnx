from typing import List, Literal, get_args

from Ne16TestConf import Ne16TestConf
from Ne16Weight import Ne16Weight
from NeurekaTestConf import NeurekaTestConf
from NeurekaWeight import NeurekaWeight
from NnxTestClasses import NnxTestConf, NnxWeight

NnxName = Literal["ne16", "neureka"]


def valid_nnx_names() -> List[str]:
    return get_args(NnxName)


def is_valid_nnx_name(name: str) -> bool:
    return name in valid_nnx_names()


def NnxWeightClsFromName(name: NnxName) -> NnxWeight:
    if name == "ne16":
        return Ne16Weight
    elif name == "neureka":
        return NeurekaWeight


def NnxTestConfClsFromName(name: NnxName) -> NnxTestConf:
    if name == "ne16":
        return Ne16TestConf
    elif name == "neureka":
        return NeurekaTestConf
