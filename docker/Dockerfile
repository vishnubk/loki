# =============================================================================
# LOKI pulsar-search container (CUDA, Apptainer-friendly)
# =============================================================================
#
# Contents
# --------
#   loki      : C++20/CUDA "Extreme Pruning" (EP) coherent pulsar search + Python
#               bindings (this repo).
#   pyloki    : Python reference implementation of the same algorithms; also a
#               RUNTIME dependency of loki (time-series IO, periodogram utils).
#   inception : pulsar/binary signal injection into filterbanks
#               (github.com/erc-compact/inception).
#   dedisp    : GPU dedispersion library (vishnubk/dedisp).
#   peasoup   : GPU acceleration search; used here mainly for its GPU
#               dedispersion + time-series dumping.
#
# Pipeline this image is built for
# --------------------------------
#   filterbank (.fil)
#     -> inception-inject            (inject synthetic pulsars / binaries)
#     -> peasoup --timeseries_dump_dir <dir> --nosearch
#                                    (GPU dedispersion, dump per-DM time series)
#     -> loki EP search on the dumped single-DM time series (.tim/.dat)
#
#   NOTE: loki itself does NOT dedisperse. Its input is one already-dedispersed
#   time series per DM trial, which is exactly what the peasoup dump produces.
#
# Build
# -----
#   # context MUST be the repo root:
#   docker build -t vishnubk/loki -f docker/Dockerfile .
#
# Convert for HPC (Apptainer/Singularity)
# ---------------------------------------
#   apptainer build loki.sif docker-daemon://vishnubk/loki:latest
#   # or, after pushing to a registry:
#   apptainer build loki.sif docker://vishnubk/loki:latest
#
# Run
# ---
#   apptainer exec --nv loki.sif python -c "import loki; print(loki.__file__)"
#   apptainer exec --nv loki.sif peasoup --help
#   apptainer exec --nv -B /data:/data loki.sif inception-inject --help
#
# =============================================================================

FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Build configuration -----------------------------------------------------
# loki / CMake syntax. "<sm>-real" emits SASS only; a bare "100" emits SASS +
# PTX so the image also runs on future (>= sm_100) GPUs via JIT.
ARG LOKI_CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100"

# dedisp / peasoup use raw nvcc flags instead of CMake.
ARG GPU_ARCH_FLAG="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_100,code=compute_100"

# Pinned upstream revisions (reproducible builds).
ARG DEDISP_COMMIT=25737499ba765dd2f48cd78c7ecf7301b440ee45
ARG PEASOUP_COMMIT=78cb72b2853b2069e19d6c03f073a51f882ea5f9
ARG PYLOKI_COMMIT=6defe14afab51c728a88ab21bf0567e405df38a6
ARG INCEPTION_COMMIT=ed0d5e64d1ca6afadfa920c6a937a97865498670

# Portable CPU baseline (upstream default is -march=native, which would bake in
# the build machine's ISA and SIGILL elsewhere). x86-64-v3 = AVX2/FMA.
ARG LOKI_MARCH=x86-64-v3
# cuRANDDx is compiled for a single SM (see cmake/FindMathDX_SM.cmake).
ARG MATHDX_SM=750
ARG JOBS=8

# --- System packages ---------------------------------------------------------
# Ubuntu 24.04 defaults to GCC 13, but loki requires GCC >= 14 for complete
# C++20 support. CUDA 12.9's nvcc accepts GCC 14 as host compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc-14 \
        g++-14 \
        git \
        ninja-build \
        wget \
        ca-certificates \
        pkg-config \
        libfftw3-dev \
        libhdf5-dev \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
        vim \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 140 \
                           --slave   /usr/bin/g++ g++ /usr/bin/g++-14 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Belt and braces: make sure every build front-end (pip, cmake, nvcc) picks
# GCC 14 even if it ignores update-alternatives.
ENV CC=gcc-14 \
    CXX=g++-14 \
    CUDAHOSTCXX=g++-14

# --- Python virtualenv -------------------------------------------------------
# The apt cmake on Ubuntu 24.04 is 3.28, but loki needs >= 3.30, so CMake comes
# from pip and the venv must precede /usr/bin on PATH.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv
RUN pip install --no-cache-dir -U pip "cmake>=3.30" ninja

WORKDIR /software

# --- dedisp ------------------------------------------------------------------
# Passing GPU_ARCH_FLAG on the make command line overrides the Makefile's
# CUDA-version autodetection (which has no sm_100 entry). Installs to /usr/local.
RUN git clone https://github.com/vishnubk/dedisp.git /software/dedisp && \
    cd /software/dedisp && \
    git checkout ${DEDISP_COMMIT} && \
    make -j${JOBS} GPU_ARCH_FLAG="${GPU_ARCH_FLAG}" && \
    make install GPU_ARCH_FLAG="${GPU_ARCH_FLAG}"

# --- peasoup -----------------------------------------------------------------
# peasoup's Makefile.inc expects DEDISP_DIR=/usr/local, satisfied above.
# "make install" copies bin/peasoup to /usr/local/bin.
# NVTX: CUDA >= 12.9 images no longer ship the legacy libnvToolsExt library
# (NVTX v3 is header-only). peasoup only uses NVTX for profiler range
# annotations, fully guarded by #ifdef USE_NVTX, so build without it.
RUN git clone https://github.com/vishnubk/peasoup.git /software/peasoup && \
    cd /software/peasoup && \
    git checkout ${PEASOUP_COMMIT} && \
    sed -i 's/ -lnvToolsExt//' Makefile && \
    make -j${JOBS} GPU_ARCH_FLAG="${GPU_ARCH_FLAG}" UCFLAGS="" && \
    make install GPU_ARCH_FLAG="${GPU_ARCH_FLAG}" UCFLAGS=""

RUN ldconfig /usr/local/lib

# --- Python scientific stack -------------------------------------------------
# scipy is an UNDECLARED runtime dependency of pyloki - install it explicitly.
RUN pip install --no-cache-dir "scipy>=1.14"

# pyloki pulls sigpyproc (from git), numba, astropy, xarray, h5py, ...
# Its build backend is uv_build; pip handles that through build isolation.
RUN pip install --no-cache-dir "git+https://github.com/pravirkr/pyloki.git@${PYLOKI_COMMIT}"

# --- loki (this repo) --------------------------------------------------------
# The pyproject's CMAKE_BUILD_PARALLEL_LEVEL "define" is ineffective; the
# environment variable is what actually controls build parallelism.
ENV CMAKE_BUILD_PARALLEL_LEVEL=${JOBS}

# Build context is the repo ROOT (see .dockerignore).
COPY . /software/loki

# CPM downloads C++ dependencies (fmt, spdlog, xsimd, HighFive, BoostMath,
# toml++, CLI11, thread-pool) from GitHub and MathDX from developer.nvidia.com
# at CMake configure time, so the build needs network access.
# CUDA::cuda_driver links against the driver stub shipped in the -devel image,
# so no GPU is required at build time.
RUN cd /software/loki && \
    pip install --no-cache-dir . \
      -C cmake.define.ENABLE_CUDA=ON \
      -C cmake.define.CMAKE_CUDA_ARCHITECTURES="${LOKI_CUDA_ARCHS}" \
      -C cmake.define.LOKI_MARCH="${LOKI_MARCH}" \
      -C cmake.define.MATHDX_SM="${MATHDX_SM}"

# --- inception (signal injection) --------------------------------------------
# inception has NO packaging (no setup.py / pyproject.toml): it is imported as a
# namespace package "injector" from the repo root placed on PYTHONPATH.
# numpy/scipy already come from the loki/pyloki install above.
# pint-pulsar is deliberately skipped (only needed for polycos mode).
RUN git clone https://github.com/erc-compact/inception.git /opt/inception && \
    cd /opt/inception && \
    git checkout ${INCEPTION_COMMIT} && \
    pip install --no-cache-dir astropy sympy pandas jplephem matplotlib

ENV PYTHONPATH=/opt/inception

RUN printf '%s\n' \
      '#!/bin/bash' \
      'exec python3 /opt/inception/injector/SCRIPT_inject_pulsars.py "$@"' \
      > /usr/local/bin/inception-inject && \
    chmod +x /usr/local/bin/inception-inject

# --- Runtime environment -----------------------------------------------------
# Apptainer runs with a read-only image and an arbitrary $HOME, so point every
# cache at /tmp and disable HDF5 file locking (breaks on many parallel FSs).
ENV MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/cache \
    HDF5_USE_FILE_LOCKING=FALSE

LABEL org.opencontainers.image.source="https://github.com/vishnubk/loki" \
      org.opencontainers.image.description="LOKI Extreme Pruning pulsar search (CUDA) with pyloki, inception, dedisp and peasoup"

WORKDIR /data

CMD ["/bin/bash"]
