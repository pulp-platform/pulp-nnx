#!/bin/bash

set -euo pipefail

# Build and install
mkdir -p $TOOLCHAIN_GNU_INSTALL_DIR
cd riscv-gnu-toolchain
./configure --prefix=$TOOLCHAIN_GNU_INSTALL_DIR --with-arch=rv32imfcxpulpv3 --with-abi=ilp32 --enable-multilib
make -j 8
