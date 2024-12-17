#!/bin/bash

set -eo pipefail

git clone https://github.com/Scheremo/pulp-sdk.git --branch scheremo --depth=1 && rm -rf pulp-sdk/.git

cd pulp-sdk

source configs/siracusa.sh
make all
