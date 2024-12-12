from enum import Enum
from typing import Dict, NamedTuple, Type

from Ne16TestConf import Ne16TestConf
from Ne16Weight import Ne16Weight
from NeurekaTestConf import NeurekaTestConf
from NeurekaV2TestConf import NeurekaV2TestConf
from NeurekaV2Weight import NeurekaV2Weight
from NeurekaWeight import NeurekaWeight
from NnxTestClasses import NnxTestConf, NnxWeight


class NnxName(Enum):
    ne16 = "ne16"
    neureka = "neureka"
    neureka_v2 = "neureka_v2"

    def __str__(self):
        return self.value


class NnxAcceleratorClasses(NamedTuple):
    testConfCls: Type[NnxTestConf]
    weightCls: Type[NnxWeight]


NnxMapping: Dict[NnxName, NnxAcceleratorClasses] = {
    NnxName.ne16: NnxAcceleratorClasses(Ne16TestConf, Ne16Weight),
    NnxName.neureka: NnxAcceleratorClasses(NeurekaTestConf, NeurekaWeight),
    NnxName.neureka_v2: NnxAcceleratorClasses(NeurekaV2TestConf, NeurekaV2Weight),
}
