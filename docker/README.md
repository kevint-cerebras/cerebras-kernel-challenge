# Docker image for Mac / Windows candidates

The Cerebras SDK runs only on Linux with Apptainer (or Singularity) installed.
This directory packages both into a Docker image so you can run the full
toolchain on any host that runs Docker (macOS, Windows, or Linux).

If you already have Linux + Apptainer, you don't need this. Follow the main
`../README.md` instead.

## Build (once, ~10 minutes, ~3 GB image)

1. Download the SDK tarball using the link in `../README.md` and extract it
   into this directory as `./sdk`:

   ```bash
   # Run from this docker/ directory.
   cp ~/Downloads/Cerebras-SDK-1.4.0-*.tar.gz .
   tar xzf Cerebras-SDK-1.4.0-*.tar.gz
   mv Cerebras-SDK-1.4.0-* sdk
   ```

   Result: `./sdk/cslc`, `./sdk/cs_python`, `./sdk/sdk-cbcore-*.sif`, etc.

2. Build:

   ```bash
   docker build --platform=linux/amd64 -t topk-knn-sdk .
   ```

   `--platform=linux/amd64` is required on Apple Silicon; the SDK is x86_64-only
   and runs under emulation on M-series Macs (functional but 2–4× slower than
   native x86).

## Use

From your submission directory (containing `layout.csl`, `run.py`, …):

```bash
docker run --rm -it --platform=linux/amd64 --privileged \
  -v "$PWD:/work" \
  topk-knn-sdk
```

Inside the container, run `cslc`, `cs_python`, `csdb`, and `pytest` as
documented in the main README and SPEC.

`--privileged` is needed for Apptainer's kernel features (namespaces, overlay,
fuse mounts) inside Docker. If your Docker host has user-namespace remapping
configured, you can drop it to `--cap-add SYS_ADMIN --device /dev/fuse`, but
most Docker Desktop setups just want `--privileged`.

## Smoke test

```bash
docker run --rm --platform=linux/amd64 --privileged topk-knn-sdk cslc --help
```

Should print `cslc` usage and exit 0. If it errors with "SIF not found" or
"singularity not in \$PATH", the image is wrong — rebuild.

## Cleanup

```bash
docker rmi topk-knn-sdk
rm -rf sdk Cerebras-SDK-1.4.0-*.tar.gz
```
