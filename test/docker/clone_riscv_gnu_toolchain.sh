#!/bin/bash

set -euo pipefail

# Clone and get all submodules except qemu
git clone https://github.com/pulp-platform/riscv-gnu-toolchain.git \
	--branch=v2.6.0 --depth=1
cd riscv-gnu-toolchain
git submodule update --init --recursive --depth=1 --recommend-shallow riscv-*
