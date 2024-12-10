#!/bin/bash

set -euo pipefail

curl -Ls https://micro.mamba.pm/api/micromamba/linux-$(uname -m)/latest | tar -xj bin/micromamba

eval "$(./bin/micromamba shell hook -s posix)"
