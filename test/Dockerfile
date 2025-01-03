ARG TOOLCHAIN_DIR=/toolchain
ARG TOOLCHAIN_GNU_INSTALL_DIR=${TOOLCHAIN_DIR}/gnu

# Build stage
FROM ubuntu:22.04 AS builder

# Needed to have the arguments in the stage
ARG TOOLCHAIN_DIR
ARG TOOLCHAIN_GNU_INSTALL_DIR

RUN DEBIAN_FRONTEND=noninteractive \
	apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y --no-install-recommends \
		autoconf \
		automake \
		autotools-dev \
		bc \
		bison \
		build-essential \
		ca-certificates \
		curl \
		cmake \
		flex \
		gawk \
		git \
		gperf \
		libexpat-dev \
		libgmp-dev \
		libmpc-dev \
		libmpfr-dev \
		libtool \
		patchutils \
		python3 \
		python3-pip \
		texinfo \
		zlib1g-dev \
	&& apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# Build gnu toolchain
ENV TOOLCHAIN_GNU_INSTALL_DIR=${TOOLCHAIN_GNU_INSTALL_DIR}
COPY docker/clone_riscv_gnu_toolchain.sh ./
RUN ./clone_riscv_gnu_toolchain.sh
COPY docker/build_riscv_gnu_toolchain.sh ./
RUN ./build_riscv_gnu_toolchain.sh

# Build pulp-sdk
COPY docker/requirements-build-pulp-sdk.txt docker/requirements-run-pulp-sdk.txt ./
RUN pip3 install --no-cache-dir -r requirements-build-pulp-sdk.txt -r requirements-run-pulp-sdk.txt
ENV PULP_RISCV_GCC_TOOLCHAIN=${TOOLCHAIN_GNU_INSTALL_DIR}
COPY docker/get_pulp_sdk.sh ./
RUN ./get_pulp_sdk.sh

# Build gvsoc
COPY docker/clone_and_build_gvsoc.sh ./
RUN ./clone_and_build_gvsoc.sh


# Minimal image
FROM ubuntu:22.04

# Needed to have the arguments in the stage
ARG TOOLCHAIN_DIR
ARG TOOLCHAIN_GNU_INSTALL_DIR

RUN DEBIAN_FRONTEND=noninteractive \
	apt-get update && \
	apt-get upgrade && \
	apt-get install -y --no-install-recommends \
		bzip2 \
		ca-certificates \
		cmake \
		curl \
		git \
		libgmp-dev \
		libmpc-dev \
		libmpfr-dev \
		make \
		python3 \
		python3-pip \
		python-is-python3 \
		zlib1g-dev \
	&& apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY requirements-pip.txt docker/requirements-run-pulp-sdk.txt ./
RUN pip3 install --no-cache-dir -r requirements-pip.txt -r requirements-run-pulp-sdk.txt --extra-index-url https://download.pytorch.org/whl/cpu

ENV PULP_SDK_HOME="/pulp-sdk"
ENV PULP_RISCV_GCC_TOOLCHAIN=${TOOLCHAIN_GNU_INSTALL_DIR}

COPY --from=builder ${TOOLCHAIN_GNU_INSTALL_DIR} ${TOOLCHAIN_GNU_INSTALL_DIR}
COPY --from=builder /pulp-sdk /pulp-sdk
COPY --from=builder /gvsoc/install /gvsoc

ENV TOOLCHAIN_GNU_INSTALL_DIR=${TOOLCHAIN_GNU_INSTALL_DIR}
ENV GVSOC="/gvsoc/bin/gvsoc"

COPY docker/requirements-run-gvsoc.txt ./
RUN pip3 install --no-cache-dir -r requirements-run-gvsoc.txt
