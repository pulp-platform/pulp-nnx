#!/bin/bash

set -euo pipefail

git clone https://github.com/lukamac/gvsoc.git \
	--depth=1 --branch=fix-vectorload --recurse-submodules --shallow-submodules
cd gvsoc
pip3 install --no-cache-dir -r requirements.txt -r gapy/requirements.txt -r core/requirements.txt
make all TARGETS=siracusa
