import os
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, Type

from NnxMapping import NnxName


class NnxBuildFlow(ABC):

    @abstractmethod
    def __init__(self, nnxName: NnxName) -> None: ...

    @abstractmethod
    def build(self) -> None: ...

    @abstractmethod
    def run(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @staticmethod
    def cmd_run(cmd: str, env=None) -> str:
        proc = subprocess.run(
            cmd.split(), check=True, capture_output=True, text=True, env=env
        )
        return proc.stdout


class MakeBuildFlow(NnxBuildFlow):
    BUILD_CMD = "make -C app all platform=gvsoc"
    RUN_CMD = "make -C app run platform=gvsoc"

    def __init__(self, nnxName: NnxName) -> None:
        self.nnxName = nnxName

    def env(self) -> os._Environ:
        _env = os.environ
        _env["ACCELERATOR"] = str(self.nnxName)
        return _env

    def build(self) -> None:
        Path("app/src/nnx_layer.c").touch()
        _ = NnxBuildFlow.cmd_run(MakeBuildFlow.BUILD_CMD, self.env())

    def run(self) -> str:
        return NnxBuildFlow.cmd_run(MakeBuildFlow.RUN_CMD, self.env())

    def __str__(self) -> str:
        return "make"


class CmakeBuildFlow(NnxBuildFlow):
    BINARY_NAME = "test-pulp-nnx"
    TOOLCHAIN_FILE = "cmake/toolchain_llvm.cmake"
    GVSOC_TARGET = "siracusa"

    def __init__(self, nnxName: NnxName) -> None:
        self.nnxName = nnxName
        self.build_dir = os.path.abspath(f"app/build_{nnxName}")
        self.gvsoc_workdir = os.path.join(self.build_dir, "gvsoc_workdir")
        assert "GVSOC" in os.environ, "The GVSOC environment variable is not set."

    def prepare(self) -> None:
        os.makedirs(self.gvsoc_workdir, exist_ok=True)
        subprocess.run(
            f"cmake -Sapp -B{self.build_dir} -GNinja -DCMAKE_TOOLCHAIN_FILE={CmakeBuildFlow.TOOLCHAIN_FILE} -DACCELERATOR={self.nnxName}".split(),
            check=True,
        )

    def build(self) -> None:
        _ = NnxBuildFlow.cmd_run(f"cmake --build {self.build_dir}")

    def run(self) -> str:
        bin = os.path.join(self.build_dir, CmakeBuildFlow.BINARY_NAME)
        gvsoc = os.environ["GVSOC"]
        cmd = f"{gvsoc} --binary {bin} --work-dir {self.gvsoc_workdir} --target {CmakeBuildFlow.GVSOC_TARGET} image flash run"
        return NnxBuildFlow.cmd_run(cmd)

    def __str__(self) -> str:
        return "cmake"


class NnxBuildFlowName(Enum):
    make = "make"
    cmake = "cmake"

    def __str__(self) -> str:
        return self.value


NnxBuildFlowClsMapping: Dict[NnxBuildFlowName, Type[NnxBuildFlow]] = {
    NnxBuildFlowName.make: MakeBuildFlow,
    NnxBuildFlowName.cmake: CmakeBuildFlow,
}
