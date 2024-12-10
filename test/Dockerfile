########## Stage 1: Large image to build toolchains and emulator ##########
FROM ubuntu:22.04 AS builder

RUN DEBIAN_FRONTEND=noninteractive \
	apt-get update && \
	apt-get upgrade && \
	apt-get install -y --no-install-recommends \
		bzip2 \
		ca-certificates \
		cmake \
		curl \
		gcc \
		git \
		git-lfs \
		g++ \
		make \
		python3 \
		python3-pip \
		python-is-python3 \
		zlib1g-dev \
	&& apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY requirements-pip.txt ./
COPY requirements-pulp-sdk.txt ./
RUN pip3 install --no-cache-dir -r requirements-pip.txt -r requirements-pulp-sdk.txt

ENV PULP_SDK_HOME="/pulp-sdk"
ENV PULP_RISCV_GCC_TOOLCHAIN="/toolchains/pulp-riscv-gnu-toolchain"

COPY _docker_pulp_riscv_gnu_toolchain.sh ./
RUN ./_docker_pulp_riscv_gnu_toolchain.sh

COPY _docker_pulp_sdk.sh ./
RUN ./_docker_pulp_sdk.sh
