#!/bin/bash

set -euo pipefail

curl -L https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2 > pulp-riscv-gnu-toolchain.tar.bz2
mkdir toolchains
tar -xvjf ./pulp-riscv-gnu-toolchain.tar.bz2 -C ./toolchains
mv ./toolchains/v1.0.16-pulp-riscv-gcc-ubuntu-18 ./toolchains/pulp-riscv-gnu-toolchain
rm ./pulp-riscv-gnu-toolchain.tar.bz2
