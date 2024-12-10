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

COPY requirements-pip.txt requirements-pulp-sdk.txt ./
RUN pip3 install --no-cache-dir -r requirements-pip.txt -r requirements-pulp-sdk.txt --extra-index-url https://download.pytorch.org/whl/cpu

ENV PULP_SDK_HOME="/pulp-sdk"
ENV PULP_RISCV_GCC_TOOLCHAIN="/toolchains/pulp-riscv-gnu-toolchain"

COPY docker/get_pulp_riscv_gnu_toolchain.sh ./
RUN ./get_pulp_riscv_gnu_toolchain.sh

COPY docker/get_pulp_sdk.sh ./
RUN ./get_pulp_sdk.sh
