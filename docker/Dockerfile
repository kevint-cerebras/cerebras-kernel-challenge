# syntax=docker/dockerfile:1
#
# Cerebras SDK 1.4.0 runtime for candidates who don't have a Linux box with
# Apptainer installed natively. Ubuntu 22.04 + Apptainer + the SDK wrappers
# and SIF copied in. See README.md in this directory for build instructions.

FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl gnupg \
      python3 python3-pip tzdata \
    && add-apt-repository -y ppa:apptainer/ppa \
    && apt-get update && apt-get install -y --no-install-recommends apptainer \
    && rm -rf /var/lib/apt/lists/*

# Apptainer bind-mounts /etc/localtime from the Docker container into the SIF
# at launch; the minimal Ubuntu image doesn't create it even with TZ set.
RUN ln -sf /usr/share/zoneinfo/UTC /etc/localtime

# The SDK wrappers (cslc, cs_python, sdk_debug_shell) invoke `singularity exec`
# directly. Apptainer is CLI-compatible, so a symlink is enough.
RUN ln -s /usr/bin/apptainer /usr/local/bin/singularity

RUN pip3 install --no-cache-dir numpy pytest

# Build context must contain ./sdk/ with the extracted SDK (cslc, cs_python,
# sdk-cbcore-*.sif, ...). See README.md for the one-liner that prepares this.
COPY sdk/ /opt/cerebras-sdk/

ENV PATH=/opt/cerebras-sdk:$PATH

WORKDIR /work
CMD ["/bin/bash"]
