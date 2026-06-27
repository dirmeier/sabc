# Reproduce the Linux CI build locally (gcc/g++ C++23 + uv).
#
# The GitHub Actions `ci` job builds the C++/MLX/nanobind extension under gcc on
# Ubuntu. Apple clang on macOS resolves some MLX overloads that gcc rejects, so
# this image lets the gcc build be verified without GitHub Actions.
#
# Build & verify:
#   docker build -t sabc-linux -f .devcontainer/Dockerfile .
#   docker run --rm -v "$PWD":/work -w /work sabc-linux \
#     bash -lc "uv sync --all-extras && make tests"
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain: gcc/g++ 13 (C++23), cmake, ninja, git, plus Python 3.13.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      g++-13 \
      gcc-13 \
      cmake \
      ninja-build \
      git \
      ca-certificates \
      curl \
      software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.13 \
      python3.13-dev \
      python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Make gcc-13/g++-13 the default compilers (C++23).
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# Install uv (the project's package/build driver).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /work

# Default to an interactive shell; CI-style verification is run explicitly via
# the `docker run ... uv sync --all-extras && make tests` command above.
CMD ["bash"]
